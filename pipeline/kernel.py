import importlib
from pathlib import Path
from .pipeline import utils

class Kernel():
    def __init__(self, competition, index):
        config_path = Path(".").resolve() / "config"
        cfg = utils.read_yaml(config_path / competition / f"{index}.yaml")
        utils.seed_everything(cfg["seed"])
        module_name = "{}Kernel".format(competition)
        print(cfg)
        """
        modulelib = importlib.import_module(module_name)
        return getattr(modulelib, module_name)(module_name, cfg)
        """
