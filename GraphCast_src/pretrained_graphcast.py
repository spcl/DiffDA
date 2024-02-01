from google.cloud.storage import Bucket
from haiku._src.data_structures import frozendict

from graphcast import checkpoint
from graphcast import graphcast
import buckets
from pathlib import Path
from typing import Sequence, Union


# # Load the Data and initialize the model

# ## Load the model params
#
# Choose one of the two ways of getting model params:
# - **random**: You'll get random predictions, but you can change the model architecture, which may run faster or fit on your device.
# - **checkpoint**: You'll get sensible predictions, but are limited to the model architecture that it was trained with, which may not fit on your device. In particular generating gradients uses a lot of memory, so you'll need at least 25GB of ram (TPUv4 or A100).
#
# Checkpoints vary across a few axes:
# - The mesh size specifies the internal graph representation of the earth. Smaller meshes will run faster but will have worse outputs. The mesh size does not affect the number of parameters of the model.
# - The resolution and number of pressure levels must match the data. Lower resolution and fewer levels will run a bit faster. Data resolution only affects the encoder/decoder.
# - All our models predict precipitation. However, ERA5 includes precipitation, while HRES does not. Our models marked as "ERA5" take precipitation as input and expect ERA5 data as input, while model marked "ERA5-HRES" do not take precipitation as input and are specifically trained to take HRES-fc0 as input (see the data section below).
#
# We provide three pre-trained models.
# 1. `GraphCast`, the high-resolution model used in the GraphCast paper (0.25 degree resolution, 37 pressure levels), trained on ERA5 data from 1979 to 2017,
#
# 2. `GraphCast_small`, a smaller, low-resolution version of GraphCast (1 degree resolution, 13 pressure levels, and a smaller mesh), trained on ERA5 data from 1979 to 2015, useful to run a model with lower memory and compute constraints,
#
# 3. `GraphCast_operational`, a high-resolution model (0.25 degree resolution, 13 pressure levels) pre-trained on ERA5 data from 1979 to 2017 and fine-tuned on HRES data from 2016 to 2021. This model can be initialized from HRES data (does not require precipitation inputs).
#


def load_model_checkpoint(gcs_bucket: Bucket, name: str) -> graphcast.CheckPoint:
    with gcs_bucket.blob(f"params/{name}").open("rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    print("Model description:\n", ckpt.description, "\n")
    print("Model license:\n", ckpt.license, "\n")

    return ckpt

def find_model_name(options: Sequence[str], resolution: float, pressure_level: int) -> Union[str, None]:
    for name in options:
        if f"resolution {resolution}" in name and f"levels {pressure_level}" in name:
            return name
    return None

def save_checkpoint(gcs_bucket: Bucket, directory: Path, name: str) -> None:
    # copy checkpoint to local disk
    with gcs_bucket.blob(f"params/{name}").open("rb") as param:
        buckets.save_to_dir(param, directory / 'params', name)
