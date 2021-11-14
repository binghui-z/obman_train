import os
import pickle

from loader.Fhb_Ho3d.hodatasets.cidata import CIdata
from loader.Fhb_Ho3d.hodatasets.ciquery import CIAdaptQueries
from loader.Fhb_Ho3d.utils.contactutils import dumped_process_contact_info


class CIOffline(CIdata):
    def __init__(
        self, data_path, hodata_path, anchor_path, hodata_use_cache=True, hodata_center_idx=9,
    ):
        super().__init__(
            data_path, hodata_path, anchor_path, hodata_use_cache=hodata_use_cache, hodata_center_idx=hodata_center_idx
        )

        # along side with basic CIDumpedQueries, we need some adapt queries
        # for offline eval
        self.queries.update(
            {
                CIAdaptQueries.OBJ_VERTS_3D,
                CIAdaptQueries.OBJ_FACES,
                CIAdaptQueries.OBJ_TSL,
                CIAdaptQueries.OBJ_ROT,
                CIAdaptQueries.HAND_VERTS_3D,
                CIAdaptQueries.HAND_JOINTS_3D,
                CIAdaptQueries.HAND_FACES,
                CIAdaptQueries.HAND_SHAPE,
                CIAdaptQueries.HAND_TSL,
                CIAdaptQueries.HAND_ROT,
                CIAdaptQueries.IMAGE_PATH,
            }
        )

    def get_dumped_processed_contact_info(self, index):
        dumped_file_path = os.path.join(self.data_path, f"{index}.pkl")
        with open(dumped_file_path, "rb") as bytestream:
            dumped_contact_info_list = pickle.load(bytestream)
        (vertex_contact, hand_region, anchor_id, anchor_elasti, anchor_padding_mask) = dumped_process_contact_info(
            dumped_contact_info_list,
            self.anchor_mapping,
            pad_vertex=self.contact_pad_vertex,
            pad_anchor=self.contact_pad_anchor,
            elasti_th=self.contact_elasti_th,
        )
        res = {
            "vertex_contact": vertex_contact,
            "hand_region": hand_region,
            "anchor_id": anchor_id,
            "anchor_elasti": anchor_elasti,
            "anchor_padding_mask": anchor_padding_mask,
        }
        return res
