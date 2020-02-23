# 🐅 Moco Image Pipeline
## 🙃 About
Pipeline codes for image tasks

---

## 🏃‍♂️ Quick Start

### Commands

```py
python ./pipeline/train.py [competition_name] [index]
```

|Competition|Name|Train|Config|
|:-|:-|:-|:-|
|[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)|`bengaliai-cv19` | [Link](https://github.com/j20232/bengali/blob/master/pipeline/Bengali.py)|[Link](https://github.com/j20232/moco_image_pipeline/tree/master/config/bengaliai-cv19")|

e.g.

```py
python ./pipeline/train.py bengaliai-cv19 0000
```

### Kaggle Kernel
To load pipeline in Kaggle kernel, you can use the following command.

[sample.py](https://github.com/j20232/moco_image_pipeline/blob/master/sample.py)
```py
from pipeline import Kernel
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

## 📈 Visualization

### MLFlow

- Remote

```
mlflow server -h [ip] -p [port]
```

- local

```
mlflow server -h 0.0.0.0 -p 8888
```

---

## 🛠 Tools

### General

- Create kaggle metadata

```py
python ./tools/create_kaggle_metadata.py [username] [competition] [id]
```

### Bengali.AI Handwritten Grapheme Classification

- Create image dataset for bengali competition|

```py
python ./tools/convert_parquet2png.py
```

---

## 📭GCP

Port Forwarding

```
gcloud compute ssh [instance_name] -- -N -f -L [local_port]:localhost:[remote_port]
```

---

## 👏 Special Thanks

- https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
- https://github.com/Ririverce/neural-network-pytorch
- https://github.com/Cadene/pretrained-models.pytorch
