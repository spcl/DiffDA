import functools
from pathlib import Path

import haiku
import xarray

import graphcast.casting as casting
import graphcast.normalization as normalization
import graphcast.autoregressive as autoregressive
from graphcast import graphcast
import haiku as hk
from graphcast.checkpoint import load

from buckets import authenticate_bucket
from demo_data import load_normalization
from pretrained_graphcast import load_model_checkpoint, find_model_name, save_checkpoint


def wrap_graphcast(model_config: graphcast.ModelConfig,
                   task_config: graphcast.TaskConfig,
                   diffs_stddev_by_level = None,
                   mean_by_level = None,
                   stddev_by_level = None,
                   wrap_autoregressive: bool = False,
                   normalize: bool = False):
    """
    Constructs and wraps the GraphCast Predictor.
    Note that this MUST be called within a haiku transform function.
    """
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    if normalize:
        assert diffs_stddev_by_level is not None
        assert mean_by_level is not None
        assert stddev_by_level is not None
        # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
        # BFloat16 happens after applying normalization to the inputs/targets.
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level)

    if wrap_autoregressive:
        # Wraps everything so the one-step model can produce trajectories.
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor
    else:
        return predictor


def retrieve_model(resolution: float,
                   pressure_levels: int,
                   model_cache_path: Path = None,
                   normalize: bool = True,
                   autoregressive: bool = True,
                   normalize_cache_path: Path = None) -> tuple[haiku.Transformed, graphcast.CheckPoint]:
    """
    Returns the haiku transformed forward function and model checkpoint for the given settings.
    The signature of the apply function is
    (params,
     rng,
     inputs: xarray.Dataset,
     targets_template: xarray.Dataset,
     forcings: xarray.Dataset
    )
    Note that there is not rng because the model is deterministic.

    Possible model values:
     - **Resolution**: 0.25deg, 1deg
     - **Pressure Levels**: 13, 37

     Not all combinations are available.
     - HRES is only available in 0.25 deg, with 13 pressure levels.

    """
    gcs_bucket = authenticate_bucket()

    # Choose the model
    params_file_options = [
        name for blob in gcs_bucket.list_blobs(prefix="params/")
        if (name := blob.name.removeprefix("params/"))]  # Drop empty string.

    params_file_name = find_model_name(params_file_options, resolution, pressure_levels)
    if params_file_name is None:
        raise FileNotFoundError(
            f"No model with given resolution ({resolution} deg) and pressure levels ({pressure_levels}) found.")

    # Load the model
    # TODO Use cached model if available
    if (model_cache_path / params_file_name).exists():
        with open((model_cache_path / params_file_name), "rb") as f:
            checkpoint = load(f, graphcast.CheckPoint)
    else:
        checkpoint = load_model_checkpoint(gcs_bucket, params_file_name)
        save_checkpoint(gcs_bucket, model_cache_path, params_file_name)
    print(checkpoint.model_config)

    diffs_stddev_by_level, mean_by_level, stddev_by_level = None, None, None
    if normalize:
        # Load normalization data
        diffs_stddev_by_level, mean_by_level, stddev_by_level = load_normalization(gcs_bucket, normalize_cache_path)

    # Build haiku transformed function
    def run_forward(inputs: xarray.Dataset,
                    targets_template: xarray.Dataset,
                    forcings: xarray.Dataset):
        predictor = wrap_graphcast(checkpoint.model_config,
                                   checkpoint.task_config,
                                   diffs_stddev_by_level,
                                   mean_by_level,
                                   stddev_by_level,
                                   wrap_autoregressive=autoregressive,
                                   normalize=normalize)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    forward = haiku.transform(run_forward)

    return forward, checkpoint