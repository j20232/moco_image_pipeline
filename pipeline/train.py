import argparse
from pathlib import Path
from pytorch_lightning import Trainer

from Bengali import Bengali
from config.bengali_config import get_cfg
import importlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Please set the index of the input config file")
    parser.add_argument("cfg_dir", help="directory of config files")
    parser.add_argument("index", help="the index of the input config file")
    cfg_dir = parser.parse_args().cfg_dir
    index = parser.parse_args().index

    # read cfg
    cfg = get_cfg()
    yaml_path = Path(".").resolve() / "config" / cfg_dir / f"{index}.yaml"
    cfg.merge_from_file(yaml_path)
    cfg.freeze()
    print(cfg)
    
    modulelib = importlib.import_module(cfg_dir)
    model = getattr(modulelib, cfg_dir)(cfg)
    print(model)

    # TODO Implement trainer
    assert False
    trainer = Trainer()
    trainer.fit(model)
