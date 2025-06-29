"""Normalization data loading utilities for GraphCast."""

import xarray
from typing import Union
from google.cloud.storage import Bucket
from xarray import Dataset


def load_normalization(gcs_bucket: Bucket,
                       cache_dir: Union[str, None] = None) -> (Dataset, Dataset, Dataset):
    def load_cached(path: str, downsample: bool = False) -> (Dataset, Dataset, Dataset):
        diffs_stddev_by_level = xarray.load_dataset(f"{path}/diffs_stddev_by_level.nc").compute()
        mean_by_level = xarray.load_dataset(f"{path}/mean_by_level.nc").compute()
        stddev_by_level = xarray.load_dataset(f"{path}/stddev_by_level.nc").compute()
        return diffs_stddev_by_level, mean_by_level, stddev_by_level

    if cache_dir is not None:
        return load_cached(cache_dir)
    else:
        with gcs_bucket.blob("stats/diffs_stddev_by_level.nc").open("rb") as f:
            diffs_stddev_by_level = xarray.load_dataset(f).compute()
        with gcs_bucket.blob("stats/mean_by_level.nc").open("rb") as f:
            mean_by_level = xarray.load_dataset(f).compute()
        with gcs_bucket.blob("stats/stddev_by_level.nc").open("rb") as f:
            stddev_by_level = xarray.load_dataset(f).compute()
        return diffs_stddev_by_level, mean_by_level, stddev_by_level