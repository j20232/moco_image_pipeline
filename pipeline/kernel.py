import importlib
from pathlib import Path
from .pipeline import utils

class Kernel():
    def __init__(self, competition, index, input_path, model_weight_path):
        config_path = Path(".").resolve() / "config"
        cfg = utils.read_yaml(config_path / competition / f"{index}.yaml")
        utils.seed_everything(cfg["seed"])
        module_name = "{}Kernel".format(competition)
        modulelib = importlib.import_module("pipeline.{}".format(module_name))
        self.model = getattr(modulelib, module_name)(cfg, input_path, model_weight_path)

    def predict(self):
        return self.model.predict()
