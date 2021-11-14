'''
Doc: https://github.com/mmatl/pyrender/blob/master/examples/example.py
pip install smplx, pyrender
'''
from pickle import TRUE
import smplx
import torch
import open3d as o3d 
import numpy as np
import trimesh
import cv2
import pyrender
import pickle
from tqdm import tqdm
from manopth.manolayer import ManoLayer
from contactopt.util import save_obj

MANO_PKL_DIR = './mano/models/'

def gui_mesh(p_verts, p_faces):
    scene = pyrender.Scene(ambient_light=np.array([0.2, 0.2, 0.2, 1.0]))
    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    spot_l = pyrender.SpotLight(color=np.ones(3), intensity=10.0,
                    innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
    point_l = pyrender.PointLight(color=np.ones(3), intensity=10.0)
    direc_l_node = scene.add(direc_l)
    spot_l_node = scene.add(spot_l)
    point_l_node = scene.add(point_l)

    hand_mesh = trimesh.Trimesh(vertices = p_verts, faces = p_faces)
    hand_mesh.fix_normals()
    mesh_id = scene.add(pyrender.Mesh.from_trimesh(hand_mesh))
    pyrender.Viewer(scene, shadows=True) # 

def gui_overlay(p_img, p_verts, p_faces, focal , princpt):
    scene = pyrender.Scene(ambient_light=np.array([0.2, 0.2, 0.2, 1.0]))
    # light
    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(direc_l, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(direc_l, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(direc_l, pose=light_pose)
    # camera
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
    # hand
    hand_mesh = trimesh.Trimesh(vertices = p_verts, faces = p_faces)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    hand_mesh.apply_transform(rot)

    mesh_id = scene.add(pyrender.Mesh.from_trimesh(hand_mesh))

    # render
    renderer = pyrender.OffscreenRenderer(viewport_width=p_img.shape[1], viewport_height=p_img.shape[0], point_size=1.0)
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    p_img = rgb * valid_mask + p_img * (1-valid_mask)

    cv2.imshow("overlay", p_img)
    cv2.waitKey()

if __name__ == "__main__":
    in_file = './data/run_opt_on_obman_test.pkl'
    anno_file = R'F:\HOinter_data\Obman\downloads\obman\test\meta\00000000.pkl'
    runs = pickle.load(open(in_file, 'rb')) 
    for idx,data in enumerate(tqdm(runs)):
        mano_poses = np.zeros(48)
        mano_poses[3:] = runs[idx]['gt_ho']['anno_poses']
        mano_pose = torch.FloatTensor([mano_poses]).view(-1,3)
        beta_in = torch.Tensor(np.zeros(10)).unsqueeze(0)

        root_pose = mano_pose[0].view(1,3)
        hand_pose = mano_pose[1:,:].view(1,-1)

        mano_layer = smplx.MANO(MANO_PKL_DIR, use_pca = False, is_rhand=True)   #default = False
        output = mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=beta_in)
        save_obj(output.vertices[0].detach().numpy(),'C:/Users/zbh/Desktop/check/'+ str(idx) +'_hand_anno_pose_smplx.obj')
        save_obj(runs[idx]['gt_ho']['anno_verts'],'C:/Users/zbh/Desktop/check/'+ str(idx) +'_hand_anno_verts.obj')
        # #gui_mesh(output.vertices[0].numpy(), mano_layer.faces)
    
        full_axang = mano_pose.view(1,-1)
        mano_model = ManoLayer(mano_root='./mano/models', use_pca=True, ncomps=45, side='right', flat_hand_mean=True)
        target_verts, target_joints = mano_model(full_axang, beta_in)
        save_obj(target_verts.squeeze(0).detach().cpu().numpy(),'C:/Users/zbh/Desktop/check/'+ str(idx) +'_hand_anno_pose_mano.obj')
        break