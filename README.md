# bengali
## About
Pipeline codes for https://www.kaggle.com/c/bengaliai-cv19/overview

## Quick Start

### Run the container

```
docker-compose build
docker-compose up -d
```

### Run commands

```
docker exec -it bengali bash
python src/main.py
```

### Commands

```
pipenv run [command] [INDEX]
```

|competition|train|test|Script|
|:-|:-|:-|:-|
|[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)|bengali_train|bengali_test|[Link](https://github.com/j20232/bengali/blob/master/pipeline/Bengali.py)|

e.g.

```
pipenv run bengali_train 0000
```

### Tools

|competition|command|meaning|
|:-|:-|:-|
|[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)|bengali_parquet|To create image dataset|


## Special Thanks

- https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
- https://github.com/Ririverce/neural-network-pytorch
