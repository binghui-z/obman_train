import argparse
from pprint import pprint
import sys
sys.path.append('/')
import numpy as np
from matplotlib import pyplot as plt
from ..loader.Fhb_Ho3d.hodatasets.hodata import HOdata

def get_fhb():
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
    return example_dataset

if __name__ == "__main__":
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
