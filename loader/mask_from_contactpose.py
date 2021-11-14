# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt

from genericpath import exists
from pickle import TRUE
import sys
import os
import argparse
from loader.ContactPose.contactpose_dataset import ContactPose,get_object_names
import loader.ContactPose.misc as mutils
import loader.ContactPose.rendering as rutils
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

def get_all_contactpose(args):
    """
    Gets all participants and objects from ContactPose
    Cuts out grasps with two hands or grasps using left hand
    :return: list of (participant_num, intent, object_name, ContactPose_object)
    """
    samples = []
    print('Reading ContactPose dataset')
    for participant_id in tqdm(range(1,args.p_num)):
        for intent in args.intent:
            print(intent)
            for object_name in get_object_names(participant_id, intent):                
                cp = ContactPose(participant_id, intent, object_name)
                """ 如果左右手都考虑的话，屏蔽这段代码 """
                # if cp._valid_hands != [1]:  # If anything else than just the right hand, remove
                #     #print(object_name)   #delete "hands"、"palm_print"   以及  handoff:bowl、utah_teapot;   use:banana、bowl、camera、ps_controller、water_bottle
                #     continue              
                samples.append((participant_id, intent, object_name, cp))

    print('Valid ContactPose samples:', len(samples))

    return samples
def create_renderers(object_name,camera_name):
    # renderer for object mesh
    # note the mesh_scale, ContactPose object models are in units of mm 
    object_renderer = rutils.DepthRenderer(object_name, cp.K(camera_name), camera_name, mesh_scale=1e-3)

    # hand renderers
    hand_renderers = []
    for mesh in cp.mano_meshes():
        if mesh is None:  # this hand is not present for this grasp
            hand_renderers.append(None)
        else:
            renderer = rutils.DepthRenderer(mesh, cp.K(camera_name), camera_name)
            hand_renderers.append(renderer)
    return {'object': object_renderer, 'hands': hand_renderers}

def show_rendering_output(renderers, color_im, camera_name, frame_idx, save_path ,crop_size=-1,write=False,vis=False):
    joints = cp.projected_hand_joints(camera_name, frame_idx)
    if crop_size > 0:
        color_im, _ = mutils.crop_image(color_im, joints, crop_size)
    
    # object rendering
    object_rendering = renderers['object'].render(cp.object_pose(camera_name, frame_idx))
    if crop_size > 0:
        object_rendering, _ = mutils.crop_image(object_rendering, joints, crop_size)    
    object_mask = object_rendering > 0
    color_im[object_mask] = (0, 255, 255)  # yellow
    
    # hand rendering
    both_hands_rendering = []
    for renderer, mask_color in zip(renderers['hands'], ((0, 255, 0), (0, 0, 255))):
        if renderer is None:  # this hand is not present for this grasp
            continue
        # hand meshes are already in the object coordinate system, so we can use
        # object pose for rendering
        rendering = renderer.render(cp.object_pose(camera_name, frame_idx))
        if crop_size > 0:
            rendering, _ = mutils.crop_image(rendering, joints, crop_size)
        both_hands_rendering.append(rendering)
        mask = rendering > 0
        color_im[mask] = mask_color
    #both_hands_rendering = np.dstack(both_hands_rendering).max(2)

    mask_result = color_im.copy()   
    object_rendering[object_rendering>0] = 255
    if len(both_hands_rendering)==1:
        both_hands_rendering[0][both_hands_rendering[0]>0] = 255
        mask_result[:,:,0] = object_rendering
        mask_result[:,:,1] = 0
        mask_result[:,:,2] = both_hands_rendering[0]
    else:
        both_hands_rendering[0][both_hands_rendering[0]>0] = 255
        both_hands_rendering[1][both_hands_rendering[1]>0] = 255
        mask_result[:,:,0] = object_rendering
        mask_result[:,:,1] = both_hands_rendering[0]
        mask_result[:,:,2] = both_hands_rendering[1]
        
    if write:
        cv2.imwrite(save_path,mask_result)

    """show""" 
    if vis:
        ##opencv show
        cv2.imshow("Masks",color_im[:, :, ::-1])
        cv2.imshow("object_mask",mask_result[:,:,0])
        cv2.imshow("hand_mask_left",mask_result[:,:,1])
        cv2.imshow("hand_mask_right",mask_result[:,:,2])
        cv2.waitKey(0)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='create mask on contactpose')
    parser.add_argument('--p_num', default=2, help='Number of folders, which the max value is 50')
    parser.add_argument('--intent',  default=["handoff","use"],type=str, choices=["use","handoff","use,handoff"])
    parser.add_argument('--object_name',default="banana", choices=["apple","banana","bowl"])
    parser.add_argument('--pen', action='store_true')
    parser.add_argument('--saveobj', action='store_true')
    parser.add_argument('--partial', default=-1, type=int, help='Only run for n samples')
    args = parser.parse_args()

    samples = get_all_contactpose(args)
    crop_size = -1   #是否进行裁剪
    for idx in range(len(samples)):
        cp = samples[idx][3]
        for camera_name in ('kinect2_left', 'kinect2_right', 'kinect2_middle'):
            image_root = os.path.join(samples[idx][3].data_tmp_dir,"images_full",camera_name,"color")
            if not exists(image_root):
                continue
            tmp = image_root.replace("images","mask")
            if not os.path.exists(tmp):
                os.makedirs(tmp)
            renderers = create_renderers(samples[idx][2],camera_name)
            for frame_idx, image_name in enumerate(tqdm(next(os.walk(image_root))[2])):
                oriImage_path = os.path.join(image_root,image_name)
                color_im = cv2.imread(oriImage_path)
                mask_path = oriImage_path.replace("images","mask")
                show_rendering_output(renderers, color_im, camera_name, frame_idx, mask_path, crop_size, write=False)