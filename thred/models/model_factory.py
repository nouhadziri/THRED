from .hred import hred_wrapper
from .topic_aware import taware_wrapper
from .thred import thred_wrapper
from .vanilla import vanilla_wrapper


def create_model(config):
    if config.type == 'vanilla':
        return vanilla_wrapper.VanillaNMTEncoderDecoder(config)
    elif config.type == 'hred':
        return hred_wrapper.HierarchicalEncoderDecoder(config)
    elif config.type == 'topic_aware':
        return taware_wrapper.TopicAwareNMTEncoderDecoder(config)
    elif config.type == 'thred':
        return thred_wrapper.TopicalHierarchicalEncoderDecoder(config)

    raise ValueError('unknown model: ' + config.type)
