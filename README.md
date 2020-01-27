# Moco Image Pipeline
## About
Pipeline codes for image tasks

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

|Competition|Train|Test|Script|Config|
|:-|:-|:-|:-|:-|
|[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)|bengali_train|**TODO**|[Link](https://github.com/j20232/bengali/blob/master/pipeline/Bengali.py)|[Link](https://github.com/j20232/bengali/tree/master/config/Bengali)|

e.g.

```
pipenv run bengali_train 0000
```

If you don't use Pipenv, you can get the definition of the commands from [Pipfile](https://github.com/j20232/bengali/blob/master/Pipfile).

### Tools

|Competition|Command|Meaning|
|:-|:-|:-|
|[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)|bengali_parquet|To create image dataset|


## Special Thanks

- https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
- https://github.com/Ririverce/neural-network-pytorch
