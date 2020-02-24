# 🐅 Moco Image Pipeline
## 🙃 About
Pipeline codes for image tasks

---

## 🏃‍♂️ Quick Start

### Commands

```py
python train.py [competition_name] [index]
```

|Competition|Name|Train|Prediction|Config|
|:-|:-|:-|:-|:-|
|[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)|`bengaliai-cv19` | [Link](https://github.com/j20232/bengali/blob/master/competition/Bengali.py) | [Link](https://github.com/j20232/bengali/blob/master/competition/BengaliKernel.py) |[Link](https://github.com/j20232/moco_image_pipeline/tree/master/config/bengaliai-cv19")|

e.g.

```py
python train.py bengaliai-cv19 0000
```

### Kaggle Kernel
To load pipeline in Kaggle kernel, you can use the following command.

[kernel_sample.py](https://github.com/j20232/moco_image_pipeline/blob/master/kernel_sample.py)

```py
from mcp import Kernel
from pathlib import Path

if __name__ == "__main__":
    # Please change here
    competition_name = "bengaliai-cv19"
    index = "0001"
    input_path = Path(".").resolve() / "input"
    model_weight_path = Path(".").resolve() / "models" / competition_name / index / f"{index}.pth"
    config_path = Path(".").resolve() / "config" / competition_name / f"{index}.yaml"
    competition_yaml_path = Path(".").resolve() / "competition.yaml"
    output_path = input_path / competition_name / "output"

    kernel = Kernel(competition_name, input_path, config_path, competition_yaml_path,
                    model_weight_path, output_path)
    kernel.predict()
```

If you want to ensemble submissions, please use `kernel.predict_for_ensemble()`.  

---

## 📈 MLFlow

```
mlflow server -h [ip] -p [port]
```

local: `ip: 0.0.0.0`, `port: 8888`

---

## 📭GCP

Port Forwarding

```
gcloud compute ssh [instance_name] -- -N -f -L [local_port]:localhost:[remote_port]
```

---

## 👏 Special Thanks

- https://github.com/Cadene/pretrained-models.pytorch
