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
pipenv run train [INDEX]
```

e.g.

```
pipenv run train 0000
```

## Special Thanks

- https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
- https://github.com/Ririverce/neural-network-pytorch
