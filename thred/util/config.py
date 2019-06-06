import codecs
import os
import shutil

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

        return model_args

    def get_infer_model_dir(self):
        for f in os.listdir(self.model_dir):
            if f == 'best_dev_ppl' and os.listdir(os.path.join(self.model_dir, f)):
                return os.path.join(self.model_dir, f)

        return self.model_dir

    def is_pretrain_enabled(self):
        return False

    def save(self):
        hparams_file = os.path.join(self.model_dir, "{}_config.yml".format(fs.file_name(self.config)))
        log.print_out("  Saving config to {}".format(hparams_file))

        to_dump_dict = dict(self.__dict__)
        if to_dump_dict['train_data']:
            to_dump_dict['train_data'] = os.path.abspath(to_dump_dict['train_data'])
        if to_dump_dict['test_data']:
            to_dump_dict['test_data'] = os.path.abspath(to_dump_dict['test_data'])
        if to_dump_dict['dev_data']:
            to_dump_dict['dev_data'] = os.path.abspath(to_dump_dict['dev_data'])
        if to_dump_dict['vocab_file']:
            to_dump_dict['vocab_file'] = os.path.abspath(to_dump_dict['vocab_file'])

        with codecs.getwriter("utf-8")(open(hparams_file, "wb")) as f:
            yaml.dump(to_dump_dict, f, default_flow_style=False)
            # f.write(json.dumps(self.__dict__, indent=2))


def _cleanup(folder):
    for f in os.listdir(folder):
        file = os.path.join(folder, f)

        if '.ckpt' in f or f.startswith('log_') or f.lower() == 'checkpoint' or f.endswith('.shuf') or f.endswith(
                '_config.yml'):
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
