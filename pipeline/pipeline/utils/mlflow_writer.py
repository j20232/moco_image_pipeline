import mlflow
from mlflow.tracking import MlflowClient

class MLflowWriter():
    def __init__(self, competition_name, index, **kwargs):
        self.client = MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(competition_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(competition_name).experiment_id
        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def log_cfg(self, cfg):
        for key, value in cfg.items():
            if type(value) is dict:
                for k, v in value.items():
                    self.client.log_param(self.run_id, "{}/{}".format(key, k), v)
            else:
                self.client.log_param(self.run_id, key, value)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, step):
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def close(self):
        self.client.set_terminated(self.run_id)
