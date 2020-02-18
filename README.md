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
|:-|:-|:-|:-|:-|
|[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)|`bengaliai-cv19` | [Link](https://github.com/j20232/bengali/blob/master/pipeline/Bengali.py)|[Link](https://github.com/j20232/bengali/tree/master/config/bengaliai-cv19")|

e.g.

```py
poetry run python ./pipeline/train.py bengaliai-cv19 0000
```

### Kaggle Kernel
To load pipeline in Kaggle kernel, you can use the following command.

```py
from pipeline import Kernel

competition_name = "bengaliai-cv19"
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
mlflow server -h 0.0.0.0 -p 8888
```

---

## 🛠 Tools

### General

- Create kaggle metadata

```py
python ./tools/create_kaggle_metadata.py [username] [competition] [id]
```

- Upload directory to Kaggle dataset

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
