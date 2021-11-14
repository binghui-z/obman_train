import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
from manopth.manolayer import ManoLayer
from scipy.spatial.distance import cdist
import pickle

from loader.DexYCB.dexycb_dataset import DexYCBDataset,read_objfile,transform_obj_verts

if __name__ == "__main__":
    split = 'all'  #  ('train', 'val', 'test', 'all') 
    seq = [0, 1, 2]
    filter_no_contact = True
    filter_thresh = 5.0
    contact_list = []
    contact_img = []
    contact_sample = []
    with open("mano/models/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces = mano_right_data["f"]

    dataset = DexYCBDataset(split, seq)
    print('Dataset size: {}'.format(len(dataset)))

    for idx in tqdm(range(82913, len(dataset))):

        sample = dataset[idx]
        label_file = sample['label_file']
        s, c, f = sample['scf']
        img_path = sample['color_file']

        # print("第 {} 序列 , 第 {} 相机, 第 {} 帧".format(s,c,f))
        # print(img_path)

        img = cv2.imread(img_path) 
        label = np.load(label_file)
        hand_shape = np.array(sample['mano_betas']).reshape(1,10)
        ycb_ids = sample['ycb_ids']  # 桌上摆放的物体的id号
        ycb_grasp_ind = sample['ycb_grasp_ind'] # 抓取物体ind
        ycb_obj_path = dataset.obj_file[ycb_ids[ycb_grasp_ind]] #  抓取物体的obj_path 
        obj_verts, obj_faces = read_objfile(ycb_obj_path) 
        joint_3d = label['joint_3d'].reshape(21, 3)
        joint_2d = label['joint_3d'].reshape(21, 3)
        seg = label['seg']   #  每个像素的lable  0 (background), 1-21 (YCB object), or 255 (hand).
        pose_objects = label['pose_y']   # [num_obj, 3, 4]
        pose_object = pose_objects[ycb_grasp_ind]
        pose_hand = label['pose_m']   # shape [1, 51]. pose_m[:, 0:48] 储存PCA表示的MANO姿态参数 ,  pose_m[0, 48:51] 储存 translation. 

        save_hand_mask = np.zeros_like(seg)
        save_hand_mask[seg == 255] = 255

        save_object_mask = np.zeros_like(seg)
        save_object_mask[seg == ycb_ids[ycb_grasp_ind]] = 255

        ################   manolay to get hand vertex    ########################

        # Load MANO layer.
        mano_layer = ManoLayer(flat_hand_mean=False,
                        ncomps=45,
                        side=sample['mano_side'],
                        mano_root='mano/models',
                        use_pca=True,return_full_pose=True)
        faces = mano_layer.th_faces.numpy()

        hand_pose = torch.from_numpy(pose_hand[:,:48].astype("float32"))
        hand_shape = torch.from_numpy(hand_shape.astype("float32"))
        hand_tran = torch.from_numpy(pose_hand[:,48:51].astype("float32"))

        # # Forward pass through MANO layer
        hand_verts, hand_joints, th_full_pose = mano_layer(hand_pose, hand_shape, hand_tran)
        hand_verts = hand_verts.cpu().numpy().squeeze(0)  * 1000
        hand_joints = hand_joints.cpu().numpy().squeeze(0) * 1000
    

        ################   get contact < 5mm seq   ########################
        if filter_no_contact:
            verts = obj_verts
            trans_verts = transform_obj_verts(verts, pose_object) * 1000

            all_dists = cdist(trans_verts, hand_joints)

            # print(all_dists.min())
            if all_dists.min() > filter_thresh:
                continue

            mano_Params = {
                'pose': hand_pose.cpu().numpy().reshape((-1)).tolist(), 
                'betas':hand_shape.reshape((-1)).tolist(), 
                'hTm': np.eye(4).reshape((-1)).tolist(), 
                'trans': hand_tran.cpu().numpy().reshape((-1)).tolist(), 
                'full_pose': th_full_pose.cpu().numpy().reshape((-1)).tolist(),
                'oTm' : pose_object.reshape((-1)).tolist(), 
            }




