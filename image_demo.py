import argparse
from copy import deepcopy
import os
import pickle
from util import render_mesh, save_obj, viewJointswObj
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from loader.Fhb_Ho3d.hodatasets.hodata import HOdata
from handobjectdatasets.queries import TransQueries, BaseQueries
from handobjectdatasets.viz2d import visualize_joints_2d_cv2

from mano_train.exputils import argutils
from mano_train.netscripts.reload import reload_model
from mano_train.visualize import displaymano
from mano_train.demo.preprocess import prepare_input, preprocess_frame


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
        default="release_models/fhb/checkpoint.pth.tar",
    )
    parser.add_argument(
        "--image_path",
        help="Path to image",
        default="readme_assets/images/color_0000.jpeg",
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

    dd = pickle.load(open("misc/mano/MANO_RIGHT.pkl", 'rb'), encoding='latin1')
    hand_face = np.array(dd['f']) 

    for i in tqdm(range(0,len(example_dataset))):

        image_path = example_dataset.image_names[i]
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

        obj_verts = flip_output["objpoints3d"].cpu().detach().numpy()[0]
        objpointscentered3d = flip_output["objpointscentered3d"].cpu().detach().numpy()[0]
        obj_faces = flip_output["objfaces"]
        obj_face_flip = obj_faces.copy()
        obj_face_flip[:,1] = obj_faces[:,2]
        obj_face_flip[:,2] = obj_faces[:,1]
        obj_trans = flip_output["objtrans"].cpu().detach().numpy()


        viewJointswObj([{"vertices": obj_verts, "faces":obj_face_flip}, 
                {"vertices": flip_verts, "faces":faces}])
        
