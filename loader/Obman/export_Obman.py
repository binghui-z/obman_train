import cv2
import json
import torch
import smplx
import trimesh
import open3d as o3d
# from utilities.import_open3d import * 
# import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle
import copy
from scipy.spatial.transform import Rotation

'''

http://www.open3d.org/docs/latest/tutorial/Basic/transformation.html
http://www.open3d.org/docs/0.12.0/tutorial/geometry/mesh.html
https://github.com/ethz-asl/maplab/blob/master/test/end-to-end-common/python/end_to_end_common/umeyama.py

Object:
iden_Dic['obj_path'].replace("/sequoia/data2/dataset/shapenet", r"F:\HOinter_data\ShapeNetCore")
 ['obj_scale'], 
 ['affine_transform'] (4,4)

Skel: 
iden_Dic['side'] left / right
iden_Dic['coords_3d'] (21, 3) hand joint position

Mano: 
iden_Dic['shape'] (10,) betas
iden_Dic['hand_pose'] (45,) hand local pose
iden_Dic['pca_pose'] (45,) hand local pose
iden_Dic['trans'] (3,) 

dict_keys(['body_tex', 'z', 'hand_depth_min', 'grasp_quality', 'obj_texture', 'obj_scale', 'coords_3d', 'sample_id', 'verts_3d', 'affine_transform', 'hand_depth_max', 'obj_depth_min', 'coords_2d', 'obj_visibility_ratio', 'grasp_epsilon', 'side', 'depth_min', 'pose', 'hand_pose', 'bg_path', 'class_id', 'obj_depth_max', 'pca_pose', 'shape', 'depth_max', 'grasp_volume', 'sh_coeffs', 'obj_path', 'trans'])
'''

MANO_PKL_DIR = "mano/models"
COLOR_LST = [[0 ,255 ,255]] +\
                    [[255, 0, 255]]*4 + [[255, 0,   0]] *4 + [[0, 255, 0]]*4 + \
                        [[255, 255, 0]]*4 + [[0, 0, 255]]*4
lines_Lst = [
            [0, 1], [1 , 2], [2 , 3], [3 , 4],
            [0, 5], [5, 6], [6 , 7], [7 , 8],
            [0, 9], [9, 10], [10,11], [11,12],
            [0,13], [13,14], [14,15], [15,16],
            [0,17], [17,18], [18,19], [19,20]
        ]

def kabsch_umeyama(tgtNC, srcNC):
    '''
    A is target, B is src
    '''
    assert tgtNC.shape == srcNC.shape
    n, m = tgtNC.shape

    EA = np.mean(tgtNC, axis=0)
    EB = np.mean(srcNC, axis=0)
    VarA = np.mean(np.linalg.norm(tgtNC - EA, axis=1) ** 2)

    H = ((tgtNC - EA).T @ (srcNC - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, t, c


def viewJointswObj(jointCN_Lst, manoDic_Lst, obj_mesh = None, maker_CN = None, with_axis = True, window_name = None):
    geo_Lst = []
    for jointCN in jointCN_Lst:
        skel_pcd = o3d.geometry.PointCloud()
        skel_pcd.points = o3d.utility.Vector3dVector(jointCN.T)
        skel_pcd.colors = o3d.utility.Vector3dVector(np.array(COLOR_LST)[:, [2,1,0]]/255.0) # to rgb, [0,1]

        skel_ls = o3d.geometry.LineSet()
        skel_ls.points = skel_pcd.points
        skel_ls.lines = o3d.utility.Vector2iVector(np.array(lines_Lst))
        geo_Lst += [skel_pcd, skel_ls]

    for manoDic in manoDic_Lst:
        vert = manoDic['vertices']
        face = manoDic['faces'] # 'joints'
        # manoDic['vertices']
        mano_mesh = o3d.geometry.TriangleMesh()
        mano_mesh.vertices = o3d.utility.Vector3dVector(vert)
        mano_mesh.triangles = o3d.utility.Vector3iVector(face)
        mano_mesh.compute_vertex_normals()
        geo_Lst += [mano_mesh]

    if obj_mesh is not None: 
        geo_Lst.append(obj_mesh)
    if maker_CN is not None: 
        mk_pcd = o3d.geometry.PointCloud()
        mk_pcd.points = o3d.utility.Vector3dVector(maker_CN.T)
        mk_pcd.colors = o3d.utility.Vector3dVector(np.array([[0,1,0] for i in range(maker_CN.shape[1])]))
        geo_Lst.append(mk_pcd)


    # o3d.io.write_triangle_mesh("hand_mesh.ply", mano_mesh)
    # o3d.io.write_triangle_mesh("obj_mesh.ply", obj_mesh)

    if with_axis: geo_Lst.append(o3d.geometry.TriangleMesh.create_coordinate_frame(0.05))
    o3d.visualization.draw_geometries(geo_Lst) # width = 640, height = 480, window_name = window_name


# ShapeNetRoot = r"\\10.193.170.132\f\HOinter_data\ShapeNetCore"
# ObManRoot = r"\\10.193.170.132\f\HOinter_data\Obman\downloads\obman\train\meta"



def saveJointswObj(savedata_dir, handname_Lst, jointCN_Lst, manoDic_Lst, manoParam_Lst, heated_mesh, maker_CN):
    '''
    handname_Lst = ['left', 'right']  -> len(jointCN_Lst) == len(manoDic_Lst) == 2
    '''
    contact_thresh = 0.4; search_r=15e-3
    for hand_idx, handname_i in enumerate(handname_Lst):
        # hand 3d joints
        joint_filename = os.path.join(savedata_dir, "joint21_" + handname_i + ".mattxt")
        np.savetxt(joint_filename, jointCN_Lst[hand_idx].T)
        # hand mono verts
        manovert_filename = os.path.join(savedata_dir, "hand_" + handname_i + ".ply")
        # np.savetxt(manovert_filename, manoDic_Lst[hand_idx]['vertices'])
        mano_mesh = o3d.geometry.TriangleMesh()
        mano_mesh.vertices = o3d.utility.Vector3dVector(manoDic_Lst[hand_idx]['vertices'])
        mano_mesh.triangles = o3d.utility.Vector3iVector(manoDic_Lst[hand_idx]['faces'])
        mano_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(manovert_filename, mano_mesh)

        # hand mono params
        manopara_filename = os.path.join(savedata_dir, "paramano_" + handname_i + ".json")
        with open(manopara_filename, "w", encoding='utf8') as fp:
            fp.write(json.dumps(manoParam_Lst[hand_idx],indent=4, ensure_ascii=False))
        # object with contact map
        object_filename = os.path.join(savedata_dir, "object" + ".ply")
        assert np.asarray(heated_mesh.vertices).shape[0] != 0, "fail to read .obj"
        o3d.io.write_triangle_mesh(object_filename, heated_mesh)

        # mano_mesh = o3d.geometry.TriangleMesh()
        # mano_mesh.vertices = o3d.utility.Vector3dVector(manoDic_Lst[hand_idx]['vertices'])
        # mano_mesh.triangles = o3d.utility.Vector3iVector(manoDic_Lst[hand_idx]['faces'])
        # mano_mesh.compute_vertex_normals()
        # handme_filename = os.path.join(savedata_dir, "hand" + ".ply")
        # o3d.io.write_triangle_mesh(handme_filename, mano_mesh)

# ShapeNetRoot = "F:\Hand_Datasets\HOinter_data"
# ObManRoot = "F:\Hand_Datasets\HOinter_data\ObMan_meta"
# saveRoot = r"F:\Hand_Datasets\HOinter_data\ObMan\Scene_inMeter"

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

ShapeNetRoot = R"F:\HOinter_data\ShapeNetCore"
ObManRoot = R"F:\HOinter_data\Obman\downloads\obman\test\meta"
saveRoot = R"F:\HOinter_data\Obman\Scene_inMeter"


if __name__ == "__main__":
    # all the instances are right hand, betas = 0, 
    mano_layer = smplx.MANO(MANO_PKL_DIR, use_pca = True, num_pca_comps = 45, is_rhand=True, flat_hand_mean = True) # right only
    '''
    important: 
    ObMan use all 45 PCA component. 
    uncomment line1583~1584  in C:\CommonPrograms\Anaconda3\envs\HPE_rebuild2021\Lib\site-packages\smplx\body_models.py:
    # if self.num_pca_comps == 45:
    #     self.use_pca = False
    '''
    for pkl_file_i in os.listdir(ObManRoot):
        iden_Dic = pickle.load(open(os.path.join(ObManRoot, pkl_file_i),'rb'),encoding='latin1')
        obj_file_name = iden_Dic['obj_path'].replace("/sequoia/data2/dataset/shapenet", ShapeNetRoot)
        print(obj_file_name)
        
        tri_mesh = as_mesh(trimesh.load_mesh(obj_file_name))
        # heated_mesh.vertices, heated_mesh.vertex_normals
        heated_mesh = o3d.geometry.TriangleMesh()
        heated_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
        heated_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
        heated_mesh.compute_vertex_normals()
        # heated_mesh = o3d.io.read_triangle_mesh()# for heated_mesh, the scale has been normed 1e-3
        

        iden_Dic['affine_transform'] # np.linalg.det(iden_Dic['affine_transform'][:3, :3])
        heated_mesh_t = copy.deepcopy(heated_mesh).transform(iden_Dic['affine_transform'])
        hTm = np.eye(4,4).astype(np.float32)
        hTm[:3, :3] = iden_Dic['affine_transform'][:3, :3] / iden_Dic['obj_scale']
        hTm[:3, 3] = iden_Dic['affine_transform'][:3, 3]
        hTm = np.linalg.inv(hTm)
        # .scale(iden_Dic['obj_scale'], center = (0,0,0))

        # hTm = iden_Dic['affine_transform']
        # hTm[:3, :3] /= iden_Dic['obj_scale']
        
        # hTm = np.linalg.inv(hTm)
        # print("check rot: ", np.linalg.det(hTm[:3, :3]))
        # .transform(iden_Dic['affine_transform'])
        # .scale(iden_Dic['obj_scale'], center = (0,0,0))
        # .transform(iden_Dic['affine_transform'])
        # norm_invTrans = heated_mesh_t.get_center()
        # heated_mesh_t = heated_mesh_t.translate(-norm_invTrans)

        # .scale(iden_Dic['obj_scale'], center = (0,0,0))
        # .transform(iden_Dic['affine_transform']
        # .scale(np.linalg.det(iden_Dic['affine_transform'][:3, :3]), center = (0,0,0))
        #.transform(iden_Dic['affine_transform']) # .scale(iden_Dic['obj_scale'], center=heated_mesh.get_center())
        # o3d.visualization.draw_geometries([heated_mesh, heated_mesh_t])

        # print("iden_Dic['pose']: \n", iden_Dic['pose']) # smplx pose
        # print("iden_Dic['hand_pose']: \n", iden_Dic['hand_pose'])
        # print("iden_Dic['pca_pose']: \n", iden_Dic['pca_pose'])
        # print("iden_Dic['trans']: \n", iden_Dic['trans'])

        # mano_layer = mano_layer_Dic[iden_Dic['side']]
        dd0 = mano_layer(
                    # betas = torch.FloatTensor(iden_Dic['shape']).view(1, -1), 
                    # global_orient = torch.FloatTensor(iden_Dic['hand_pose']).view(1,3), 
                    hand_pose = torch.FloatTensor(iden_Dic['pca_pose']).view(1,-1), 
                    # transl = torch.FloatTensor(iden_Dic['trans']).view(1, -1), 
                    return_full_pose = True)

        global_R, global_t, global_c = kabsch_umeyama(
                        iden_Dic['coords_3d'][[0,1, 5,9,13,17]], 
                        dd0.joints[0].detach().numpy()[[0, 13, 1, 4, 10, 7]]
                            )
        # hTm[:3, 3] += global_t
        trans_Arr = global_t - np.dot(np.eye(3)-global_R, dd0.joints[0].detach().numpy()[0])
        rvec_Arr = (Rotation.from_matrix(global_R)).as_rotvec()
        dd = mano_layer(
                    # betas = torch.FloatTensor(iden_Dic['shape']).view(1, -1), 
                    transl = torch.FloatTensor(trans_Arr).view(1, -1), 
                    global_orient = torch.FloatTensor(rvec_Arr).view(1,3), 
                    hand_pose = torch.FloatTensor(iden_Dic['pca_pose']).view(1,-1), 
                    return_full_pose = True)

        # (iden_Dic['coords_3d']).T,
        # manoDic = {'vertices':  iden_Dic['verts_3d'], 'faces': mano_layer.faces} # norm_invTrans
        
        # mano_vert = dd.vertices[0].detach().numpy()
        mano_vert = iden_Dic['verts_3d']
        # manoDic_para = {'vertices': np.dot(mano_vert , hTm[:3, :3].T) + hTm[:3, 3], 'faces': mano_layer.faces}
        mano_Dic = {'vertices': np.dot(mano_vert , hTm[:3, :3].T) + hTm[:3, 3], 'faces': mano_layer.faces }
        mano_joint21 = np.dot(iden_Dic['coords_3d'] , hTm[:3, :3].T)  + hTm[:3, 3]
        # viewJointswObj([ mano_joint21.T ], [mano_Dic], heated_mesh_t.transform(hTm), iden_Dic['coords_3d'][[0,1, 5,9,13,17]].T) # norm_invTrans

        mano_Params = {
            'pose': rvec_Arr.tolist() + iden_Dic['pca_pose'].tolist(), 
            'betas':iden_Dic['shape'].tolist(), 
            'hTm': hTm.reshape((-1)).tolist(), 
            'trans': trans_Arr.tolist(), 
            'full_pose': dd.full_pose[0].numpy().tolist()
        }
        cur_saveDir = "train_" + pkl_file_i.split(".")[0] + "_" + iden_Dic['class_id'] + "_" + iden_Dic['sample_id'] + "_1"
        cur_saveDir = os.path.join(saveRoot, cur_saveDir)
        if not os.path.isdir(cur_saveDir):
                os.mkdir(cur_saveDir)

        saveJointswObj(cur_saveDir, ['right'], [mano_joint21], 
                    [mano_Dic], 
                    [mano_Params], 
                    heated_mesh_t.transform(hTm), 
                    None
                    )