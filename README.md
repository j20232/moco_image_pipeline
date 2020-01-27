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

### Create images

```
pipenv run parquet
```

### Train a model

```
pipenv run train
```

## Special Thanks

- https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
