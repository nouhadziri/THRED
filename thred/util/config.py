import os
import shutil
from pathlib import Path

import yaml

from . import fs, log


class Config(dict):

    def __init__(self, *args, **kwargs) -> None:
        super(Config, self).__init__(*args, **kwargs)

        margs = self.__read_params()
        self.update(margs)

        pargs = dict(kwargs)
        empty_args = [arg for arg, val in pargs.items() if arg in margs and val is None]
        for arg in empty_args:
            pargs.pop(arg)

        # For same parameters, model arguments from config file are overwritten by program arguments
        self.update(pargs)
        self.__dict__ = self

    def __read_params(self):
        if self['mode'] == 'train':
            config_file = None

            if self['config'] is None and os.path.exists(self['model_dir']):
                for f in os.listdir(self['model_dir']):
                    if f.endswith('_config.yml'):
                        config_file = os.path.join(self['model_dir'], f)

            if config_file is None:
                missing_args = []

                if self['train_data'] is None:
                    missing_args.append('train_data')

                if self['dev_data'] is None:
                    missing_args.append('dev_data')

                if self['config'] is None:
                    missing_args.append('config')

                if missing_args:
                    raise ValueError('In train mode, the following arguments are required: {}'.format(
                        ', '.join(missing_args)))

                config_file = self['config']

            if not os.path.exists(self['model_dir']):
                os.makedirs(self['model_dir'])
            elif self['restart_training']:
                _cleanup(self['model_dir'])
        else:
            if not os.path.exists(self['model_dir']):
                raise ValueError('model directory does not exist')

            config_file = None
            for f in os.listdir(self['model_dir']):
                if f.endswith('_config.yml'):
                    config_file = os.path.join(self['model_dir'], f)

            if not config_file:
                raise ValueError('config file not found in model directory')

        with open(config_file, 'r') as file:
            model_args = yaml.safe_load(file)

        self._update_relative_paths(model_args)

        return model_args

    def get_infer_model_dir(self):
        for f in os.listdir(self.model_dir):
            if f == 'best_dev_ppl' and os.listdir(os.path.join(self.model_dir, f)):
                return os.path.join(self.model_dir, f)

        return self.model_dir

    def is_pretrain_enabled(self):
        return False

    def save(self):
        hparams_file = Path(self.model_dir) / "{}_config.yml".format(fs.file_name(self.config))
        log.print_out("  Saving config to {}".format(hparams_file))

        config_dict = dict(self.__dict__)

        # absolute paths
        if config_dict['train_data']:
            config_dict['train_data'] = Path(config_dict['train_data']).absolute().as_posix()
        if config_dict['test_data']:
            config_dict['test_data'] = Path(config_dict['test_data']).absolute().as_posix()
        if config_dict['dev_data']:
            config_dict['dev_data'] = Path(config_dict['dev_data']).absolute().as_posix()

        # relative paths
        if config_dict['vocab_file']:
            config_dict['vocab_file'] = Path(config_dict['vocab_file']).name
        if config_dict['vocab_pkl']:
            config_dict['vocab_pkl'] = Path(config_dict['vocab_pkl']).name
        if config_dict['checkpoint_file']:
            config_dict['checkpoint_file'] = Path(config_dict['checkpoint_file']).name
        if config_dict.get('topic_vocab_file', ''):
            config_dict['topic_vocab_file'] = Path(config_dict['topic_vocab_file']).name
        if config_dict.get('best_dev_ppl_dir', ''):
            config_dict['best_dev_ppl_dir'] = Path(config_dict['best_dev_ppl_dir']).name

        with hparams_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def _update_relative_paths(self, args):
        model_path = Path(self["model_dir"])
        if "vocab_file" in args:
            args["vocab_file"] = (model_path / args["vocab_file"]).as_posix()

        if "vocab_pkl" in args:
            args["vocab_pkl"] = (model_path / args["vocab_pkl"]).as_posix()

        if "checkpoint_file" in args:
            args["checkpoint_file"] = (model_path / args["checkpoint_file"]).as_posix()

        if "topic_vocab_file" in args:
            args["topic_vocab_file"] = (model_path / args["topic_vocab_file"]).as_posix()

        if "best_dev_ppl_dir" in args:
            args["best_dev_ppl_dir"] = (model_path / args["best_dev_ppl_dir"]).as_posix()


def _cleanup(folder):
    for f in os.listdir(folder):
        file = os.path.join(folder, f)

        if '.ckpt' in f or f.startswith('log_') or \
                f.lower() == 'checkpoint' or \
                f.endswith('.shuf') or \
                f.endswith('_config.yml'):
            os.remove(file)
        elif (f.endswith('_log') or f.startswith('best_')) and os.path.isdir(file):
            shutil.rmtree(file)

    log.print_out("  >> Heads up: model directory cleaned up!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    config = Config(vars(args))
    config.save()
