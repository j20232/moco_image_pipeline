import argparse
import importlib
import warnings
from pathlib import Path
from pipeline.utils import seed_everything, read_yaml


def main():
    # argparser
    parser = argparse.ArgumentParser(description="Please set the index of the input config file")
    parser.add_argument("competition", help="directory of config files")
    parser.add_argument("index", help="the index of the input config file")
    parser.add_argument('-w', '--show_warnings', action='store_false',
                        help="whether to show debug messages")
    competition = parser.parse_args().competition
    index = parser.parse_args().index
    config_path = Path(".").resolve() / "config"
    cfg = read_yaml(config_path / competition / f"{index}.yaml")

    # initialization
    if parser.parse_args().show_warnings:
        warnings.simplefilter("ignore")
    seed_everything(cfg["seed"])

    # training
    modulelib = importlib.import_module(competition)
    classifier = getattr(modulelib, competition)(competition, index, cfg)
    classifier.fit()


if __name__ == "__main__":
    main()
