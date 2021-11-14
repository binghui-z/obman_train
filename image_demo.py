import argparse
from copy import deepcopy
import os
import pickle

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import json
from handobjectdatasets.queries import TransQueries, BaseQueries
from handobjectdatasets.viz2d import visualize_joints_2d_cv2
import open3d as o3d
from mano_train.exputils import argutils
from mano_train.netscripts.reload import reload_model
from mano_train.visualize import displaymano
from mano_train.demo.preprocess import prepare_input, preprocess_frame
from util import saveJointswObj, viewJointswObj


def forward_pass_3d(model, input_image, pred_obj=True):
    sample = {}
    sample[TransQueries.images] = input_image
    sample[BaseQueries.sides] = ["left"]
    sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float()
    sample["root"] = "wrist"
    if pred_obj:
        sample[TransQueries.objpoints3d] = input_image.new_ones(
            (1, 600, 3)
        ).float()
    _, results, _ = model.forward(sample, no_loss=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint",
        default="release_models/obman/checkpoint.pth.tar",
    )
    parser.add_argument(
        "--image_path",
        help="Path to image",
        default=R"readme_assets/images/can.jpg",
        #default=R"C:/Users/zbh/Desktop/core50/1.png",
    )
    parser.add_argument(
        "--no_beta", action="store_true", help="Force shape to average"
    )
    args = parser.parse_args()
    argutils.print_args(args)

    checkpoint = os.path.dirname(args.resume)
    with open(os.path.join(checkpoint, "opt.pkl"), "rb") as opt_f:
        opts = pickle.load(opt_f)

    # Initialize network
    model = reload_model(args.resume, opts, no_beta=args.no_beta)

    model.eval()

    print(
        "Input image is processed flipped and unflipped "
        "(as left and right hand), both outputs are displayed"
    )

    # load faces of hand
    with open("misc/mano/MANO_LEFT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces = mano_right_data["f"]
    a = pickle.load(open(R"F:\HOinter_data\core50\labels2names.pkl",'rb'))
    fig = plt.figure(figsize=(4, 4))
    fig.clf()
    #default=R"readme_assets/images/can.jpg",
    #default=R"C:/Users/zbh/Desktop/core50/1.png",
    image_path = R"C:/Users/zbh/Desktop/core50/10.png"
    save_dir = R'C:\Users\zbh\Desktop\core50\10'
    frame = cv2.imread(image_path)
    frame = preprocess_frame(frame)
    input_image = prepare_input(frame)
    img = Image.fromarray(frame.copy())
    hand_crop = cv2.resize(np.array(img), (256, 256))

    noflip_hand_image = prepare_input(hand_crop, flip_left_right=False)
    flip_hand_image = prepare_input(hand_crop, flip_left_right=True)
    noflip_output = forward_pass_3d(model, noflip_hand_image)
    flip_output = forward_pass_3d(model, flip_hand_image)
    flip_verts = flip_output["verts"].cpu().detach().numpy()[0]*1000
    noflip_verts = noflip_output["verts"].cpu().detach().numpy()[0]*1000
# param type verts

    hand_verts = flip_output["verts"].cpu().detach().numpy()[0]*1000
    pose = flip_output["pose"].cpu().detach().numpy()[0]
    shape = flip_output["shape"].cpu().detach().numpy()[0]
    fullpose = flip_output["fullpose"].cpu().detach().numpy()[0]
    joints3d = flip_output["joints"].cpu().detach().numpy()[0]
    obj_verts = flip_output["objpoints3d"].cpu().detach().numpy()[0]
    objpointscentered3d = flip_output["objpointscentered3d"].cpu().detach().numpy()[0]
    obj_faces = flip_output["objfaces"]
    obj_face_flip = obj_faces.copy()
    obj_face_flip[:,1] = obj_faces[:,2]
    obj_face_flip[:,2] = obj_faces[:,1]
    obj_trans = flip_output["objtrans"].cpu().detach().numpy()
###########
    mano_Dic = {'vertices': hand_verts , 'faces': faces }
    object_Dic = {'vertices': obj_verts , 'faces': obj_face_flip }
    mano_joint21 = joints3d*1000
    mano_Params = {
        'pose': pose.reshape((-1)).tolist(), 
        'betas':shape.reshape((-1)).tolist(), 
        'hTm': np.eye(4).reshape((-1)).tolist(), 
        'fullpose':fullpose.reshape((-1)).tolist(), 
        'obj_trans': obj_trans.reshape((-1)).tolist(), 
        }
    viewJointswObj([ mano_joint21.T ],[{"vertices": obj_verts, "faces":obj_face_flip}, 
             {"vertices": hand_verts, "faces":faces}])
    saveJointswObj(save_dir, ['right'], [mano_joint21],
                    [mano_Dic], 
                    [mano_Params], 
                    [object_Dic], 
                    None
                    )

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    ax.title.set_text("flipped input")
    displaymano.add_mesh(ax, flip_verts, faces, flip_x=True)
    if "objpoints3d" in flip_output:
        objverts = flip_output["objpoints3d"].cpu().detach().numpy()[0]
        displaymano.add_mesh(
            ax, objverts, flip_output["objfaces"], flip_x=True, c="r"
        )
    flip_inpimage = deepcopy(np.flip(hand_crop, axis=1))
    if "joints2d" in flip_output:
        joints2d = flip_output["joints2d"]
        flip_inpimage = visualize_joints_2d_cv2(
            flip_inpimage, joints2d.cpu().detach().numpy()[0]
        )
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(np.flip(flip_inpimage[:, :, ::-1], axis=1))

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.title.set_text("unflipped input")
    displaymano.add_mesh(ax, noflip_verts, faces, flip_x=True)
    if "objpoints3d" in noflip_output:
        objverts = noflip_output["objpoints3d"].cpu().detach().numpy()[0]
        displaymano.add_mesh(
            ax, objverts, noflip_output["objfaces"], flip_x=True, c="r"
        )
    noflip_inpimage = deepcopy(hand_crop)
    if "joints2d" in flip_output:
        joints2d = noflip_output["joints2d"]
        noflip_inpimage = visualize_joints_2d_cv2(
            noflip_inpimage, joints2d.cpu().detach().numpy()[0]
        )
    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(np.flip(noflip_inpimage[:, :, ::-1], axis=1))
    plt.show()