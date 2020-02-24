import os
import sys
import argparse
import importlib
import warnings
from pathlib import Path

from mcp import utils
sys.path.append(os.path.join("./competition"))


def main():
    # argparser
    parser = argparse.ArgumentParser(description="Please set the index of the input config file")
    parser.add_argument("competition", help="directory of config files")
    parser.add_argument("index", help="the index of the input config file")
    parser.add_argument("-l", "--train_on_local", action="store_true",
                        help="whether to train on the local instance")
    parser.add_argument('-w', '--show_warnings', action='store_false',
                        help="whether to show debug messages")
    competition = parser.parse_args().competition
    index = parser.parse_args().index
    config_path = Path(".").resolve() / "config"
    cfg = utils.read_yaml(config_path / competition / f"{index}.yaml")
    competitions = utils.read_yaml(Path(".").resolve() / "competition.yaml")

    # initialization
    if parser.parse_args().show_warnings:
        warnings.simplefilter("ignore")
    utils.seed_everything(cfg["seed"])

    # training
    yml_competition_name = competitions[competition]
    modulelib = importlib.import_module(yml_competition_name)
    classifier = getattr(modulelib, yml_competition_name)(competition, index, cfg,
                                                          is_local=parser.parse_args().train_on_local)
    classifier.fit()


if __name__ == "__main__":
    main()
