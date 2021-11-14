import argparse
import pickle
from pprint import pprint
import sys

import cv2

from contactopt.util import render_mesh, save_obj, viewJointswObj, vis_2d_pose, vis_Joints3D, vis_mesh
from loader.Fhb_Ho3d.hodatasets.fhbutils import transform_obj_verts
sys.path.append('/')
import numpy as np
from matplotlib import pyplot as plt
from loader.Fhb_Ho3d.hodatasets.hodata import HOdata
from tqdm import tqdm
import os
from manopth.manolayer import ManoLayer
import torch

example_dataset = HOdata.get_dataset(
        dataset="fhbhands",
        data_root="F:\HOinter_data",
        data_split="train",
        split_mode="actions",
        use_cache=True,
        mini_factor=1.0,
        center_idx=9,
        enable_contact=True,
        filter_no_contact=True,
        filter_thresh=5.0,
        synt_factor=1,
        like_v1=False,
    )
print("length of fhb is:",len(example_dataset))
root = R'F:\HOinter_data\fhbhands\procress_dataset'

dd = pickle.load(open("mano/models/MANO_RIGHT.pkl", 'rb'), encoding='latin1')
hand_face = np.array(dd['f']) 

for i in tqdm(range(0,len(example_dataset))):
    element = example_dataset.sample_infos[i]
    data_root = os.path.join(root, element['subject'], element['action_name'], element['seq_idx'],"meta")
    mask_root = os.path.join(root, element['subject'], element['action_name'], element['seq_idx'],"mask")
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    if not os.path.exists(mask_root):
        os.makedirs(mask_root)
    pkl_path = example_dataset.image_names[i].replace("Video_files_480","procress_dataset").replace("jpeg","pkl").replace("color","meta")
    mask_path = example_dataset.image_names[i].replace("Video_files_480","procress_dataset").replace("color","mask")
##### hand param
    anno = {}
    meta = {}
    anno['image_name'] = example_dataset.image_names[i]
    anno['mano_infos'] = example_dataset.mano_infos[i]
    anno["cam_extr"] = np.array(example_dataset.cam_extr,dtype=np.float32)
    anno['cam_intr'] = np.array(example_dataset.cam_intr,dtype=np.float32)

    pose = torch.from_numpy(anno['mano_infos']['fullpose']).unsqueeze(0).float()
    shape = torch.from_numpy(anno['mano_infos']['shape']).unsqueeze(0).float()

    side = anno['mano_infos']["side"]
    ncomps = anno['mano_infos']["ncomps"]
    if pose.shape[1] == 48:   # Special case when we're loading GT honnotate
        mano_model = ManoLayer(mano_root='./mano/models', joint_rot_mode="axisang",use_pca=False, ncomps=ncomps, side=side, flat_hand_mean=True)
    else:   # Everything else
        mano_model = ManoLayer(mano_root='./mano/models', use_pca=True, ncomps=ncomps, side=side, flat_hand_mean=False)
    
    verts, joints = mano_model(pose, shape)
    joints_numpy = joints.squeeze(0).detach().cpu().numpy() + anno['mano_infos']['trans']
    verts_numpy = verts.squeeze(0).detach().cpu().numpy()+ anno['mano_infos']['trans']

##### object param
    anno['object_name'] = example_dataset.objnames[i] 
    anno['obj_verts'] = example_dataset.split_objects[anno['object_name']]['verts']
    anno['obj_faces'] = example_dataset.split_objects[anno['object_name']]['faces']
    anno['obj_trans'] = example_dataset.objtransforms[i]

##### project
    joints_reproj = joints_numpy.dot(anno['cam_intr'].T)
    joints_reproj = (joints_reproj / joints_reproj[:, [2]])[:, :2].astype(np.float32)

    verts_reproj = verts_numpy.dot(anno['cam_intr'].T)
    verts_reproj = (verts_reproj / verts_reproj[:, [2]])[:, :].astype(np.float32)

    obj_verts_trans = transform_obj_verts(anno['obj_verts'],anno['obj_trans'],anno["cam_extr"])
    anno['obj_verts_trans'] = obj_verts_trans
    obj_verts_reproj = obj_verts_trans.dot(anno['cam_intr'].T)
    obj_verts_reproj = (obj_verts_reproj / obj_verts_reproj[:, [2]])[:, :].astype(np.float32)

#### save param
    meta["joints_img"] = joints_reproj
    meta["joints_mano"] = joints_numpy
    meta["verts_mano"] = verts_numpy
    meta["obj_verts_trans"] = obj_verts_trans
    meta["obj_verts_img"] = obj_verts_reproj
    meta["obj_faces"] = anno['obj_faces']
    meta["cam_extr"] = anno["cam_extr"]
    meta["cam_intr"] = anno["cam_intr"]
    # f =  open(pkl_path, 'wb')
    # pickle.dump(meta,f)

####### GUI (obj trans using our code )########
    # obj_posed_verts = anno['obj_verts'].dot(anno['obj_trans'][:3, :3].T) + anno['obj_trans'][:3, 3]/1000
    # viewJointswObj([joints_numpy.T], [{"vertices": obj_posed_verts, "faces":anno['obj_faces']}, 
    #     {"vertices": verts_numpy, "faces":hand_face}])
###############

#### display
    img = cv2.imread(anno['image_name'])
    mesh_img = vis_mesh(img,np.concatenate((verts_reproj[:,:2],obj_verts_reproj[:,:2]),axis=0))
    cv2.imshow("mesh_result",mesh_img)

    joint_img  = vis_2d_pose(img,joints_reproj)
    cv2.imshow("joint_result",joint_img)
    
    mask_result = img.copy()
    camera_param= np.eye(4)
    _,_,render_R,render1 = render_mesh(img,verts_reproj,hand_face,camera_param = camera_param)
    _,_,render_O,render2 = render_mesh(img,obj_verts_reproj,face=anno['obj_faces'],camera_param = camera_param)
    cv2.imshow("render1",render1/255 )
    cv2.imshow("render2",render2/255 )

    if side == 'right':
        mask_result[:,:,0] = render_O
        mask_result[:,:,1] = 0
        mask_result[:,:,2] = render_R
    else:
        mask_result[:,:,0] = render_O
        mask_result[:,:,1] = render_R
        mask_result[:,:,2] = 0
    #cv2.imwrite(mask_path,mask_result)
    cv2.imshow("MASK0",mask_result )
    cv2.waitKey(0)

