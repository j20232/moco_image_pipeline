# 🐅 Moco Image Pipeline

![](https://github.com/j20232/moco_image_pipeline/blob/master/assets/logo.png)

PyTorch-based image pipeline for kaggle competition

---

## 🎵 Installation

```
git clone --recursive git@github.com:j20232/moco_image_pipeline.git
pip install -r requirements.txt
```

---

## 🏃‍♂️ Quick Start

### Training

```py
python train.py [competition_name] [index]
```

|Competition|Train|Kernel|Config|
|:-|:-|:-|:-|
|[`bengaliai-cv19`](https://www.kaggle.com/c/bengaliai-cv19)|[Link](https://github.com/j20232/moco_image_pipeline/blob/master/competition/Bengali.py) | [Link](https://github.com/j20232/moco_image_pipeline/blob/master/competition/BengaliKernel.py) |[Link](https://github.com/j20232/moco_image_pipeline/tree/master/config/bengaliai-cv19)|

e.g.

```py
python train.py bengaliai-cv19 0000
```

### Submission at Kaggle Kernel
To load pipeline in Kaggle kernel, you can use this module as follows.

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

---

## 📭 Others

- Port forwarding to GCP instances

```
gcloud compute ssh [instance_name] -- -N -f -L [local_port]:localhost:[remote_port]
```

- mlflow (local: `ip: 0.0.0.0`, `port: 8888`)

```
mlflow server -h [ip] -p [port]
```

- test

```
python setup.py test
```

## ➕ To Add Competitions

1. Create the competition class file at https://github.com/j20232/moco_image_pipeline/tree/master/competition
2. Add correspondence between the competition name and class to https://github.com/j20232/moco_image_pipeline/blob/master/competition.yaml

---

## 🐣 Tested Models

### Official Implementation

Reference: https://pytorch.org/docs/stable/torchvision/models.html

- `resnet18`, `resnet34`, `resnet50`, resnet101`, `resnet152`
- `resnext50_32x4d`, `resnext101_32x4d`
- `densenet121`, `densenet169`, `densenet201`, `densenet161` 
- `mobilenet_v2`
- `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`

### pretrained-models

Reference: https://github.com/Cadene/pretrained-models.pytorch

- `resnext101_64x4d`
- `nasnetalarge`,
- `nasnetamobile`
- `dpn68`, `dpn68b`, `dpn92`, `dpn98`, `dpn131`, `dpn107`
- `xception`
- `senet154`, `se_resnet50`, `se_resnet101`, `se_resnet152`, `se_resnext50_32x4d`, `se_resnext101_32x4d`
- `pnasnet5large`

### pytorch-image-models

Reference: https://github.com/rwightman/pytorch-image-models

- `mnasnet_050`, `mnasnet_075`, `mnasnet_100`, `mnasnet_140`
- `semnasnet_050`, `semnasnet_075`, `semnasnet_100`, `semnasnet_140`
- `mnasnet_small`
- `mobilenetv2_100`
- `fbnetc_100`
- `spnasnet_100`
- `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b2a`
- `efficientnet_b3`, `efficientnet_b3a`, `efficientnet_b4`, `efficientnet_b5`
- `efficientnet_b6`, `efficientnet_b7`, `efficientnet_b8`
- `efficientnet_es`, `efficientnet_em`, `efficientnet_el`
- `efficientnet_cc_b0_4e`, `efficientnet_cc_b0_8e`, `efficientnet_cc_b1_8e`
- `mixnet_s`, `mixnet_m`, `mixnet_l`, `mixnet_xl`, `mixnet_xxl`
