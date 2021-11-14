import argparse
from pprint import pprint
import sys
sys.path.append('/')
import numpy as np
from matplotlib import pyplot as plt
from loader.Fhb_Ho3d.hodatasets.hodata import HOdata

example_dataset = HOdata.get_dataset(
        dataset="ho3d",
        data_root="F:/HOinter_data/HO3D_v3",
        data_split="trainval",  #choices = [("train", "trainval"), "val", "test"]
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
print("end")