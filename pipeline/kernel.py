import importlib
from pathlib import Path
from .pipeline import utils


class Kernel():
    def __init__(self, competition, index, input_path, weight_path):
        config_path = Path(".").resolve() / "config"
        cfg = utils.read_yaml(config_path / competition / f"{index}.yaml")
        utils.seed_everything(cfg["seed"])
        competition_dict = utils.read_yaml(Path(".").resolve() / "competition.yaml")
        module_name = "{}Kernel".format(competition_dict[competition])
        modulelib = importlib.import_module("pipeline.{}".format(module_name))
        self.model = getattr(modulelib, module_name)(competition, cfg, input_path, weight_path)

    def predict(self):
        return self.model.predict()
