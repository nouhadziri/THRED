from models.hred import hred_wrapper
from models.topic_aware import taware_wrapper
from models.vanilla import vanilla_wrapper
from models.thred import thred_wrapper


def create_model(config):
    if config.type == 'vanilla':
        return vanilla_wrapper.VanillaNMTEncoderDecoder(config)
    elif config.type == 'hred':
        return hred_wrapper.HRED(config)
    elif config.type == 'topic_aware':
        return taware_wrapper.TopicAwareNMTEncoderDecoder(config)
    elif config.type == 'thred':
        return thred_wrapper.THRED(config)

    raise ValueError('unknown model: ' + config.type)
