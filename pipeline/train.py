import argparse
import importlib
from pytorch_lightning import Trainer
import mlflow
import mlflow.pytorch


def main():
    parser = argparse.ArgumentParser(description="Please set the index of the input config file")
    parser.add_argument("cfg_dir", help="directory of config files")
    parser.add_argument("index", help="the index of the input config file")
    parser.add_argument('-e', "--epochs", default=1, help="epochs")
    parser.add_argument('-c', '--use_cpu', action='store_true', 
                        help="whether to use gpu")

    cfg_dir = parser.parse_args().cfg_dir
    index = parser.parse_args().index
    modulelib = importlib.import_module(cfg_dir)
    model = getattr(modulelib, cfg_dir)(cfg_dir, index)

    # training
    if parser.parse_args().use_cpu:
        trainer = Trainer(max_epochs=parser.parse_args().epochs, train_percent_check=0.1)
    else:
        trainer = Trainer(max_epochs=parser.parse_args().epochs, gpus=[0])
    trainer.fit(model)

    # logging
    # with mlflow.start_run() as run:
        # params
        # mlflow.log_param("debug_param", 0)

        # results
        # mlflow.log_metric("debug_metric", 100)
        

if __name__ == "__main__":
    main()
