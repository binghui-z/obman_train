import pickle
import sys
from contactopt.util import render_mesh, viewJointswObj, vis_2d_pose
sys.path.append('/')
import numpy as np
from loader.Fhb_Ho3d.hodatasets.hodata import HOdata
from tqdm import tqdm
import os
from manopth.manolayer import ManoLayer
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

def SortJoints(ORIarray):
    """
    ORIarray: original input
    return: the output after sorting
    """
    NEWarray = ORIarray.copy()
    link = [0,1,2,3,]
    for i in range(len(NEWarray)):
        NEWarray[i] = ORIarray[link[i]]
    return NEWarray

if __name__ ==  "__main__":
    example_dataset = HOdata.get_dataset(
            dataset="ho3d",
            data_root="F:/HOinter_data/HO3D_v3",
            data_split="val",  #choices = [("train", "trainval"), "val", "test"]
            split_mode="paper",  #choices = ["paper" , "objects","official"]
            use_cache=True,
            mini_factor=1.0,
            center_idx=9,
            enable_contact=True,
            filter_no_contact=True,
            filter_thresh=5.0,
            synt_factor=1,
            like_v1=True,
        )
    print("length of fhb is:",len(example_dataset))

    dd = pickle.load(open("mano/models/MANO_RIGHT.pkl", 'rb'), encoding='latin1')
    hand_face = np.array(dd['f'])       
    
    process_root = R"F:\HOinter_data\HO3D_v3\HO3D_v3\process_dataset"
    data_root = R'F:\HOinter_data\HO3D_v3\HO3D_v3\train'
    for i in tqdm(range(0,len(example_dataset))):
        meta = {}
        element = example_dataset.idxs[i]
        image_path = example_dataset.seq_map[element[0]][element[1]]['img'].replace('png','jpg')
        
        if not os.path.exists(image_path):
            print(image_path)
            continue
        tmp1 = os.path.join(process_root,element[0],"mask")
        if not os.path.exists(tmp1):
            os.makedirs(tmp1)
        tmp2 = os.path.join(process_root,element[0],"param")
        if not os.path.exists(tmp2):
            os.makedirs(tmp2)
        mask_path = image_path.replace("rgb","mask").replace("train","process_dataset")
        param_path = image_path.replace("rgb","param").replace("jpg","pkl").replace("train","process_dataset")

        mano_param = example_dataset.seq_map[element[0]][element[1]]
#### hand param
        pose = mano_param['handPose']   #1*48
        shape = mano_param['handBeta']  #1*10
        trans = mano_param['handTrans'] #1*3
        joints3d = mano_param['handJoints3D'] #21*3

#### object param 
        objRot = mano_param['objRot']  #1*3
        objTrans = mano_param['objTrans']  #1*3

        obj_name = mano_param['objName']
        obj_verts = example_dataset.obj_meshes[obj_name]['verts']
        obj_faces = example_dataset.obj_meshes[obj_name]['faces']
#### camera param
        camera_extr = np.array(example_dataset.cam_extr)
        camera_intr = mano_param['camMat']   #3*3

        theta = torch.from_numpy(pose).unsqueeze(0).float()
        beta = torch.from_numpy(shape).unsqueeze(0).float()
        trans_RT = torch.from_numpy(trans).unsqueeze(0).float()
        if len(pose) == 48:   # Special case when we're loading GT honnotate
            mano_model = ManoLayer(mano_root='./mano/models', joint_rot_mode="axisang",use_pca=False, ncomps=45, side='right', flat_hand_mean=True)
        else:   # Everything else
            mano_model = ManoLayer(mano_root='./mano/models', use_pca=True, ncomps=15, side='right', flat_hand_mean=False)
        
        verts, joints = mano_model(theta, beta)
        joints_numpy = joints.squeeze(0).detach().cpu().numpy() + trans.T
        handverts_numpy = verts.squeeze(0).detach().cpu().numpy() + trans.T

#### project 
        joint_project = (camera_extr[:3,:3].dot(joints_numpy.T).T).dot(camera_intr.T)  
        joint_project = (joint_project / joint_project[:, [2]])[:, :].astype(np.float32)
        
        hand_vert_project = (camera_extr[:3,:3].dot(handverts_numpy.T).T).dot(camera_intr.T)  
        hand_vert_project = (hand_vert_project / hand_vert_project[:, [2]])[:, :].astype(np.float32)

        # try:
        #     r = R.from_rotvec(objRot) 
        # except ValueError:
        #     r = R.from_rotvec(np.array(objRot.T))
        # rot = r.as_matrix()
        rot = cv2.Rodrigues(objRot)[0]
        objvert_numpy = rot.dot(obj_verts.transpose()).transpose() + objTrans
        trans_verts = camera_extr[:3, :3].dot(objvert_numpy.transpose()).transpose()
        obj_vert_tmp = np.array(trans_verts).astype(np.float32)
        point3d = np.array(camera_intr).dot(obj_vert_tmp.T).T
        obj_vert_project = (point3d / point3d[:, 2:])[:, :].astype(np.float32)

        # objvert_numpy = obj_verts.dot(rot.T) + objTrans
        # obj_vert_project = (camera_extr[:3,:3].dot(objvert_numpy.T).T).dot(camera_intr.T) 
        # obj_vert_project = (obj_vert_project / obj_vert_project[:, [2]])[:, :].astype(np.float32)

        meta["joints_img"] = joint_project
        meta["joints_mano"] = joints_numpy
        meta["verts_mano"] = handverts_numpy
        meta["obj_verts_trans"] = objvert_numpy
        meta["obj_verts_img"] = obj_vert_project
        meta["obj_faces"] = obj_faces
        meta["cam_extr"] = camera_extr
        meta["cam_intr"] = camera_intr
        #f =  open(param_path, 'wb')
        #pickle.dump(meta,f)
### display
        viewJointswObj([joints_numpy.T], [{"vertices": objvert_numpy, "faces":obj_faces}, 
            {"vertices": handverts_numpy, "faces":hand_face}])
        oriImage = cv2.imread(image_path)
        mask_result = oriImage.copy()
        _,_,mask_handR,render1 = render_mesh(oriImage,hand_vert_project,hand_face,camera_param=np.eye(3))
        _,_,mask_object,render2 = render_mesh(oriImage,obj_vert_project,obj_faces,camera_param=np.eye(3))
        cv2.imshow("render1",render1/255)
        cv2.imshow("render2",render2/255)

        mask_result[:,:,0] = mask_object
        mask_result[:,:,1] = 0
        mask_result[:,:,2] = mask_handR
        #cv2.imwrite(mask_path,mask_result)
        cv2.imshow("mask",mask_result)

        image_joint = vis_2d_pose(oriImage,joint_project[:,:2])
        cv2.imshow("joint",image_joint)
        cv2.waitKey(-1)

        
    