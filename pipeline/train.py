import argparse
import re
import yaml
import importlib
import warnings
import mlflow
import mlflow.pytorch
from pathlib import Path
from pipeline.utils.seed import seed_everything

ROOT_PATH = Path(".").resolve()
CONFIG_PATH = ROOT_PATH / "config"


def main():
    parser = argparse.ArgumentParser(description="Please set the index of the input config file")
    parser.add_argument("cfg_dir", help="directory of config files")
    parser.add_argument("index", help="the index of the input config file")
    parser.add_argument('-w', '--show_warnings', action='store_false',
                        help="whether to show debug messages")

    cfg_dir = parser.parse_args().cfg_dir
    index = parser.parse_args().index
    if parser.parse_args().show_warnings:
        warnings.simplefilter("ignore")

    # load a config file
    yaml_path = CONFIG_PATH / cfg_dir / f"{index}.yaml"
    loader = yaml.SafeLoader    # to read float number like 1e-5
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(yaml_path) as f:
        cfg = yaml.load(f, Loader=loader)
    seed_everything(cfg["seed"])

    # training
    modulelib = importlib.import_module(cfg_dir)
    classifier = getattr(modulelib, cfg_dir)(cfg)
    model, best_results, final_epoch = classifier.fit()

    # logging
    with mlflow.start_run():
        mlflow.log_param("Competition", cfg_dir)
        mlflow.log_param("index", index)
        for key, value in cfg.items():
            if type(value) is dict:
                for k, v in value.items():
                    mlflow.log_param("{} - {}".format(key, k), v)
            else:
                mlflow.log_param(key, value)
        mlflow.log_param("final_epoch", final_epoch)
        mlflow.log_metrics(best_results)
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
