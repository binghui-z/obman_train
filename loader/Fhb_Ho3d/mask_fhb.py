import argparse
import pickle
from pprint import pprint
import sys
import open3d as o3d
import cv2

from contactopt.util import render_mesh, save_obj, saveJointswObj, viewJointswObj, vis_2d_pose, vis_Joints3D, vis_mesh
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

for i in tqdm(range(4384,len(example_dataset))):
    element = example_dataset.sample_infos[i]
    data_root = os.path.join(root, element['subject'], element['action_name'], element['seq_idx'],"meta")
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    pkl_root = example_dataset.image_names[i].replace("Video_files_480","procress_dataset").replace("jpeg","pkl").replace("color","meta")
    
##### hand param
    anno = {}
    anno['image_name'] = example_dataset.image_names[i]
    #anno['joints2d'] = example_dataset.joints2d[i]
    #anno['joints3d'] = example_dataset.joints3d[i]
    anno['mano_infos'] = example_dataset.mano_infos[i]
    anno["cam_extr"] = np.array(example_dataset.cam_extr,dtype=np.float32)
    anno['cam_intr'] = np.array(example_dataset.cam_intr,dtype=np.float32)
    # anno['kp_2d'] = example_dataset.get_hand_verts2d(i)
    # get_hand_axisang_wrt_cam
    fullpose = anno['mano_infos']['fullpose']
    pose = torch.from_numpy(anno['mano_infos']['pose']).unsqueeze(0).float()
    shape = torch.from_numpy(anno['mano_infos']['shape']).unsqueeze(0).float()
    trans = torch.from_numpy(anno['mano_infos']['trans']).unsqueeze(0).float()
    side = anno['mano_infos']["side"]
    ncomps = anno['mano_infos']["ncomps"]
    if pose.shape[1] == 48:   # Special case when we're loading GT honnotate
        #mano_model = ManoLayer(mano_root='./mano/models', joint_rot_mode="axisang", use_pca=True, center_idx=None, flat_hand_mean=True)
        mano_model = ManoLayer(mano_root='./mano/models', joint_rot_mode="axisang",use_pca=True, ncomps=ncomps, side=side, flat_hand_mean=True)
    else:   # Everything else
        mano_model = ManoLayer(mano_root='./mano/models', use_pca=True, ncomps=ncomps, side=side, flat_hand_mean=False)
    verts, joints = mano_model(pose, shape, trans)
    joints_numpy = joints.squeeze(0).detach().cpu().numpy()
    verts_numpy = verts.squeeze(0).detach().cpu().numpy()
    anno['joints3d'] = joints_numpy
    anno['verts'] = verts_numpy
    ###############
    joints_reproj = joints_numpy.dot(anno['cam_intr'])
    #joints_reproj = (joints_reproj / joints_reproj[:, [2]])[:, :2].astype(np.int32)

    verts_reproj = verts_numpy.dot(anno['cam_intr'].T)
    #verts_reproj = (verts_reproj / verts_reproj[:, [2]])[:, :].astype(np.int32)

    ###############
    anno['object_name'] = example_dataset.objnames[i] 
    anno['obj_verts'] = example_dataset.split_objects[anno['object_name']]['verts']
    anno['obj_faces'] = example_dataset.split_objects[anno['object_name']]['faces']
    anno['obj_trans'] = example_dataset.objtransforms[i]
    #transform_obj_verts(anno['obj_verts'],anno['obj_trans'],anno["cam_extr"])
    # anno['obj_verts_trans'] = obj_verts_trans

    obj_posed_verts = anno['obj_verts'].dot(anno['obj_trans'][:3, :3].T) + anno['obj_trans'][:3, 3]/1000
    viewJointswObj([joints_numpy.T], [{"vertices": obj_posed_verts, "faces":anno['obj_faces']}, 
        {"vertices": verts_numpy, "faces":hand_face}])
    #obj_verts_reproj = obj_verts_trans.dot(anno['cam_intr'].T)
    #obj_verts_reproj = (obj_verts_reproj / obj_verts_reproj[:, [2]])[:, :].astype(np.int32)

    mano_Params = {
    'pose': pose.numpy().reshape((-1)).tolist(), 
    'betas':shape.reshape((-1)).tolist(), 
    'hTm': anno['obj_trans'].reshape((-1)).tolist(), 
    'trans': trans.numpy().reshape((-1)).tolist(), 
    'full_pose': fullpose.reshape((-1)).tolist()
    }
    mano_joint21 = joints_numpy
    mano_Dic = {'vertices': verts_numpy , 'faces': hand_face }
    object_Dic = {'vertices': obj_posed_verts, 'faces': anno['obj_faces'] }
    
    temp_dir = R'C:\Users\zbh\Desktop\fhb'
    save_dir = os.path.join(temp_dir,element['subject'],element['action_name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saveJointswObj(save_dir, ['right'], [mano_joint21], 
            [mano_Dic], 
            [mano_Params], 
            [object_Dic], 
            None
            )
    print(example_dataset.image_names[i])
