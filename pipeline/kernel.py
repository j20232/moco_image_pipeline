import importlib
from .pipeline import utils


class Kernel():
    def __init__(self, competition, input_path, config_path,
                 competition_yaml_path, weight_path, output_path):
        cfg = utils.read_yaml(config_path)
        utils.seed_everything(cfg["seed"])
        competition_dict = utils.read_yaml(competition_yaml_path)
        module_name = "{}Kernel".format(competition_dict[competition])
        modulelib = importlib.import_module("pipeline.{}".format(module_name))
        self.model = getattr(modulelib, module_name)(competition, cfg, input_path,
                                                     weight_path, output_path)

    def predict(self):
        return self.model.predict()
