import xarray
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from tqdm import tqdm
import wandb

from graphcast import graphcast, normalization, casting, xarray_jax, xarray_tree
from graphcast.data_utils import get_day_progress, get_year_progress, featurize_progress

def get_forcing(time: xarray.DataArray, lon: xarray.DataArray, timesteps: np.ndarray, num_timesteps: int, batch_size: int = 0, forcing_type: str = "diffusion") -> None:

    DAY_PROGRESS = "day_progress"
    YEAR_PROGRESS = "year_progress"

    # Compute seconds since epoch.
    # Note `data.coords["datetime"].astype("datetime64[s]").astype(np.int64)`
    # does not work as xarrays always cast dates into nanoseconds!
    batch_dim = ("batch",)
    seconds_since_epoch = time.data.astype("datetime64[s]").astype(np.int64)
    seconds_since_epoch = seconds_since_epoch.reshape((batch_size, 1))

    # Add year progress features.
    year_progress = get_year_progress(seconds_since_epoch)
    forcing_dict = {}
    forcing_dict.update(featurize_progress(
            name=YEAR_PROGRESS, dims=batch_dim + ("time",), progress=year_progress))
    # Add day progress features.
    day_progress = get_day_progress(seconds_since_epoch, lon.data)
    forcing_dict.update(featurize_progress(
            name=DAY_PROGRESS,
            dims=batch_dim + ("time",) + lon.dims,
            progress=day_progress))
    
    # hijack year_progress_sin for timesteps
    if forcing_type == "diffusion":
        forcing_dict["year_progress_sin"].data = (timesteps / num_timesteps * 2 - 1).astype(np.float32).reshape(forcing_dict["year_progress_sin"].shape)

    ds_forcing = xarray.Dataset(forcing_dict).drop_vars(["day_progress", "year_progress"])

    return ds_forcing


def wrap_graphcast(model_config: graphcast.ModelConfig,
                   task_config: graphcast.TaskConfig,
                   stddev_by_level: xarray.Dataset,
                   mean_by_level: xarray.Dataset,
                   diffs_stddev_by_level: xarray.Dataset,):
    """
    Constructs and wraps the GraphCast Predictor.
    Note that this MUST be called within a haiku transform function.
    """
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResidualsForDiffusion(predictor, 
                                                             stddev_by_level=stddev_by_level, 
                                                             mean_by_level=mean_by_level, 
                                                             diffs_stddev_by_level=diffs_stddev_by_level)
    return predictor

def wrap_graphcast_prediction(model_config: graphcast.ModelConfig,
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

    return predictor

def _to_numpy_xarray(array) -> xarray.Dataset:
    # Unwrap
    vars = xarray_jax.unwrap_vars(array)
    coords = xarray_jax.unwrap_coords(array)
    # Ensure it's numpy
    vars = {n: (array[n].dims, np.asarray(v)) if len(array[n].dims) > 0 else np.asarray(v)for n,v in vars.items()}
    coords = {n: (array[n].dims, np.asarray(v)) if len(array[n].dims) > 0 else np.asarray(v) for n, v in coords.items()}
    # Create new dataset
    copied_dataset = xarray.Dataset(vars, coords)
    return copied_dataset

def _to_jax_xarray(dataset: xarray.Dataset, device) -> xarray_jax.Dataset:
    # Unwrap
    vars = dataset.variables
    coords = dataset.coords
    # Ensure it's numpy
    vars = {n: (dataset[n].dims, jax.device_put(jnp.array(v.data), device)) if len(dataset[n].dims) > 0 else jax.device_put(jnp.array(v.data), device) for n,v in vars.items()}
    coords = {n: (dataset[n].dims, jax.device_put(jnp.array(v.data), device)) if len(dataset[n].dims) > 0 else jax.device_put(jnp.array(v.data), device) for n,v in coords.items()}
    # Create new dataset
    copied_dataset = xarray_jax.Dataset(vars, coords)
    return copied_dataset

def validation_step(forward_fn: hk.TransformedWithState, norm_original_fn, norm_diff_fn, params, state, validate_dataset, args, device, rng, mode="ddpm"):
    pbar = enumerate(validate_dataset)
    progress_bar = False
    if args.rank == 0:
        pbar = tqdm(pbar, desc="Validation", total=len(validate_dataset))
        progress_bar = True
    for batch_idx, batch in pbar:
        rng_batch = jax.random.fold_in(rng, batch_idx)
        timesteps = np.ones((args.validation_batch_size,), dtype=np.int32)
        inputs_pred = batch['graphcast']
        inputs_ground_truth = batch['weatherbench']
        inputs_static = batch['static']
        datetime = inputs_ground_truth.datetime
        lon = inputs_ground_truth.lon
        norm_forcings = get_forcing(datetime, lon, timesteps, args.num_train_timesteps, batch_size=args.validation_batch_size)
        norm_forcings = _to_jax_xarray(norm_forcings, device)
        inputs_pred = _to_jax_xarray(inputs_pred.drop_vars("datetime"), device)
        inputs_ground_truth = _to_jax_xarray(inputs_ground_truth.drop_vars("datetime"), device)
        inputs_static = _to_jax_xarray(inputs_static, device)
        norm_soa = norm_original_fn(inputs_ground_truth["toa_incident_solar_radiation"])
        norm_forcings = xarray.merge([norm_forcings, norm_soa])
        norm_inputs_pred = norm_original_fn(inputs_pred)
        if 'total_precipitation_6hr' in norm_inputs_pred.data_vars and args.resolution == "0.25deg":
            norm_inputs_pred = norm_inputs_pred.drop_vars('total_precipitation_6hr')
        norm_static = norm_original_fn(inputs_static)
        if mode == "ddpm":
            corrected_pred, _ = forward_fn.apply(params, state, None, norm_inputs_pred, norm_forcings, norm_static, rng_batch, progress_bar=progress_bar)
        elif mode == "repaint":
            mask = batch["mask"]
            measurements_interp = batch["weatherbench_interp"].drop_vars(["datetime", "toa_incident_solar_radiation"])
            mask = _to_jax_xarray(mask, device)["mask"] # convert from dataset to dataarray
            measurements_interp = _to_jax_xarray(measurements_interp, device)
            measurements_diff_interp = measurements_interp - inputs_pred
            norm_measurements_diff_interp = norm_diff_fn(measurements_diff_interp)
            corrected_pred, _ = forward_fn.apply(params, state, None, mask = mask,
                                                    norm_measurements_diff_interp = norm_measurements_diff_interp,
                                                    norm_inputs_pred = norm_inputs_pred,
                                                    norm_forcings = norm_forcings,
                                                    norm_static = norm_static,
                                                    rng_batch = rng_batch,
                                                    progress_bar = True,)
        else:
            raise ValueError(f"Unknown mode {mode}")

        diff = corrected_pred - inputs_ground_truth
        diff_gc = inputs_pred - inputs_ground_truth

        diff_500hPa = diff.sel(level=500)
        diff_gc_500hPa = diff_gc.sel(level=500)

        val_loss = {f"val/{mode}/diffusion_rmse_500hPa/{k}": jnp.sqrt(xarray_jax.unwrap_data((v*v).mean())).item() for k,v in diff_500hPa.data_vars.items()}
        val_loss_gc = {f"val/{mode}/graphcast_rmse500hPa/{k}": jnp.sqrt(xarray_jax.unwrap_data((v*v).mean())).item() for k,v in diff_gc_500hPa.data_vars.items()}
        #val_loss["rank"] = args.rank

        if args.rank == 0:
            pbar.set_postfix(val_loss_z500=val_loss[f"val/{mode}/diffusion_rmse_500hPa/geopotential"], loss_gc_z500=val_loss_gc[f"val/{mode}/graphcast_rmse500hPa/geopotential"])
            if args.use_wandb:
                log_dict = {**val_loss, **val_loss_gc}
                wandb.log(log_dict)