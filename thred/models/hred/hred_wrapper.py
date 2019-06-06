from ..hierarchical_base import BaseHierarchicalEncoderDecoder
from . import hred_helper


class HierarchicalEncoderDecoder(BaseHierarchicalEncoderDecoder):
    def __init__(self, config):
        super(HierarchicalEncoderDecoder, self).__init__(config)

    def _get_model_helper(self):
        return hred_helper

    def _get_checkpoint_name(self):
        return "hred"
