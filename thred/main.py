import tensorflow as tf
import logging

from .models import model_factory
from .util.config import Config

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)


def main(_):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=('train', 'interactive', 'test'), help='work mode')
    parser.add_argument('--model_dir', type=str, required=True, help='model directory')

    parser.add_argument('--config', type=str, help='config file containing parameters to configure the model')

    parser.add_argument('--train_data', type=str, help='training dataset')
    parser.add_argument('--dev_data', type=str, help='development dataset')
    parser.add_argument('--test_data', type=str, help='tests dataset')

    parser.add_argument('--embed_conf', type=str, default="conf/word_embeddings.yml", help='embedding config file')
    parser.add_argument('--restart_training', action='store_true', help='remove saved models and logs in the model directory to start training from scratch')
    parser.add_argument('--eval_best_model', action='store_true', help='whether to evaluate the best model after training finished')

    parser.add_argument('--num_gpus', type=int, default=4, help='number of GPUs to use')
    parser.add_argument('--n_responses', type=int, default=1, help='number of generated responses')
    parser.add_argument('--beam_width', type=int, help='beam width to override the value in config file')
    parser.add_argument('--length_penalty_weight', type=float,
                        help='length penalty to override the value in config file')
    parser.add_argument('--sampling_temperature', type=float,
                        help='sampling temperature to override the value in config file')
    parser.add_argument('--lda_model_dir', type=str, help='required only for testing with topical models (THRED and TA-Seq2Seq)')

    args = vars(parser.parse_args())
    config = Config(**args)

    model = model_factory.create_model(config)

    if config.mode == 'train':
        model.train()
    elif config.mode == 'interactive':
        model.interactive()
    elif config.mode == 'test':
        model.test()


if __name__ == "__main__":
    tf.app.run()
