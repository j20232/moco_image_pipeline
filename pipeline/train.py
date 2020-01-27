import argparse
from pathlib import Path
from pytorch_lightning import Trainer

from bengali_module import BengaliModule
from config.bengali_config import get_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Please set the index of the input config file")
    parser.add_argument("index", help="the index of the input config file")
    index = parser.parse_args().index
    assert len(index) == 4

    # read cfg
    cfg = get_cfg()
    yaml_path = Path(".").resolve() / "config" / f"{index}.yaml"
    cfg.merge_from_file(yaml_path)
    cfg.freeze()
    print(cfg)

    model = BengaliModule(cfg)

    # TODO Implement trainer
    assert False
    trainer = Trainer()
    trainer.fit(model)
