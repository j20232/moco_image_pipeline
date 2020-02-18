# 🐅 Moco Image Pipeline
## 🙃 About
Pipeline codes for image tasks

---

## 🏃‍♂️ Quick Start

### Commands

```
pipenv run [command] [INDEX]
```

|Competition|Name|Train|Script|Config|
|:-|:-|:-|:-|:-|
|[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)|`Bengali` | `bengali_train`| [Link](https://github.com/j20232/bengali/blob/master/pipeline/Bengali.py)|[Link](https://github.com/j20232/bengali/tree/master/config/Bengali)|

e.g.

```
pipenv run bengali_train 0000
```

If you don't use Pipenv, you can get the definition of the commands from [Pipfile](https://github.com/j20232/bengali/blob/master/Pipfile).

### Kaggle Kernel
To load pipeline in Kaggle kernel, you can use the following command.

```py
from pipeline import Kernel

competition_name = "Bengali"
index = "0001"
kernel = Kernel(competition_name, index)
```

---

## 🐳 Docker
### Run the container

```
docker-compose build
docker-compose up -d
```
** Now debugging **

### Run commands

```
docker exec -it bengali bash
python src/main.py
```

** Now debugging **

---

## 📈 Visualization

### MLFlow

- Remote

```
mlflow server -h [ip] -p [port]
```

- local

```
pipenv run local_mlflow
```

---

## 🛠 Tools

|Competition|Command|Meaning|
|:-|:-|:-|
|General|`dataset [COMPETITION] [ID]`|Create metadata.json for Kaggle dataset at local env.|
|General|`upload`|Upload `./upload` directory to Kaggle dataset|
|[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)|`bengali_parquet`|Create image dataset for bengali competition|

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
