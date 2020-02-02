import argparse
import json
from pathlib import Path

UPLOAD_DIR = Path(".").resolve() / "upload"
META = "dataset-metadata.json"

p = argparse.ArgumentParser(description="")
p.add_argument("user_id", help="your id")
p.add_argument("title", help="directory of config files")
p.add_argument("id", help="the index of the input config file")
p.add_argument("-l", "--license", default="CC0-1.0", help="license of your dataset")

meta_data = {
    "title": p.parse_args().title,
    "id": "{}/{}".format(p.parse_args().user_id, p.parse_args().id),
    "licenses": [{"name": p.parse_args().license}]
}

f = open(str(UPLOAD_DIR / META), "w")
json.dump(meta_data, f, indent=4)

# TODO zip here
