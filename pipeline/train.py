import argparse
from pytorch_lightning import Trainer
import importlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Please set the index of the input config file")
    parser.add_argument("cfg_dir", help="directory of config files")
    parser.add_argument("index", help="the index of the input config file")
    cfg_dir = parser.parse_args().cfg_dir
    index = parser.parse_args().index
    modulelib = importlib.import_module(cfg_dir)
    model = getattr(modulelib, cfg_dir)(cfg_dir, index)
    print(model)

    # TODO Implement trainer
    assert False
    trainer = Trainer()
    trainer.fit(model)
