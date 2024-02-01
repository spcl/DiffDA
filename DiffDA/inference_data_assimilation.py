from argparse import ArgumentParser
from tqdm import tqdm
import wandb
import os
from pathlib import Path
import functools

import jax
import jax.numpy as jnp

#rank = int(os.environ.get('JAX_RANK', '0'))
#jax.config.update("jax_default_device", jax.devices()[rank])

import numpy as np
import haiku as hk
import xarray
from graphcast import xarray_jax, normalization
from diffusers import FlaxDDPMScheduler
from datetime import datetime
import pandas as pd

from checkpoint import load_checkpoint as load_diffusion_checkpoint
from diffusion_common import get_forcing, _to_jax_xarray, _to_numpy_xarray
from diffusion_common import wrap_graphcast as wrap_graphcast_diffusion, wrap_graphcast_prediction
from dataloader import load_normalization, load_model_checkpoint, GraphCastDiffusionDataset

PRESSURE_LEVEL_VARS=["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind", "vertical_velocity", "specific_humidity"]
SURFACE_LEVEL_VARS=["2m_temperature", "mean_sea_level_pressure", "10m_v_component_of_wind", "10m_u_component_of_wind", "total_precipitation_6hr"]
SURFACE_LEVEL_VARS_NO_TP=["2m_temperature", "mean_sea_level_pressure", "10m_v_component_of_wind", "10m_u_component_of_wind"]
PRESSURE_LEVELS=[  50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925, 1000]



def calculate_stat_rmse(diff: xarray.Dataset, data_type: str):
    drop_tp = "total_precipitation_6hr" not in diff.data_vars
    if drop_tp:
        surface_vars = SURFACE_LEVEL_VARS_NO_TP
    else:
        surface_vars = SURFACE_LEVEL_VARS
    data_pl_dict = {var : np.array(jnp.sqrt(xarray_jax.unwrap_data((diff[var]*diff[var]).mean(dim=["lon", "lat"])))).ravel() for var in PRESSURE_LEVEL_VARS}
    data_sl_dict = {var : np.array(jnp.sqrt(xarray_jax.unwrap_data((diff[var]*diff[var]).mean(dim=["lon", "lat"])))).ravel() for var in surface_vars}
    data_pl_dict["data_type"] = data_type
    data_sl_dict["data_type"] = data_type
    df_pl = pd.DataFrame(data_pl_dict, index=PRESSURE_LEVELS)
    df_sl = pd.DataFrame(data_sl_dict, index=[0])
    df_pl["level"] = df_pl.index
    df_sl["level"] = 0
    return df_pl, df_sl

def autoregressive_assimilation(graphcast_fn: hk.TransformedWithState, repaint_6h_fn: hk.TransformedWithState, repaint_48h_fn: hk.TransformedWithState, 
                                norm_original_fn, norm_diff_fn, graphcast_params, repaint_6h_params, repaint_48h_params, validate_dataset, args, device, rng):
    pbar = tqdm(range(args.num_autoregressive_steps), desc="Validation", total=len(validate_dataset))
    assert len(validate_dataset) >= args.num_autoregressive_steps
    dataset_iter = iter(validate_dataset)
    validation_batch_size = 1

    corrected_pred, corrected_pred_prev = None, None
    inputs_pred, inputs_pred_prev = None, None

    graphcast_fn_jitted = jax.jit(graphcast_fn.apply)
    
    for batch_idx in pbar:
        rng_batch = jax.random.fold_in(rng, batch_idx)
        timesteps = np.ones((validation_batch_size,), dtype=np.int32)
        batch = next(dataset_iter)
        
        inputs_ground_truth = batch['weatherbench']
        inputs_static = batch['static']
        datetime = inputs_ground_truth.datetime
        lon = inputs_ground_truth.lon
        if batch_idx <= 1:
            # only load gc prediction at first step, use previous prediction for the rest
            inputs_pred_prev = inputs_pred
            inputs_pred = batch['graphcast']
            inputs_pred = _to_jax_xarray(inputs_pred.drop_vars("datetime"), device)
            repaint_fn = repaint_48h_fn
            num_train_timesteps = args.num_train_timesteps_48h
            repaint_params = repaint_48h_params
        else:
            repaint_fn = repaint_6h_fn
            num_train_timesteps = args.num_train_timesteps_6h
            repaint_params = repaint_6h_params
        forcings = get_forcing(datetime, lon, timesteps, num_train_timesteps, batch_size=validation_batch_size)
        forcings_prediction = get_forcing(datetime, lon, timesteps, num_train_timesteps, batch_size=validation_batch_size, forcing_type="prediction")
        forcings = _to_jax_xarray(forcings, device)
        forcings_prediction = _to_jax_xarray(forcings_prediction, device)
        inputs_ground_truth = _to_jax_xarray(inputs_ground_truth.drop_vars("datetime"), device)
        inputs_static = _to_jax_xarray(inputs_static, device)
        toa = inputs_ground_truth["toa_incident_solar_radiation"]
        norm_toa = norm_original_fn(toa)
        norm_forcings = xarray.merge([forcings, norm_toa])
        norm_forcings_prediction = xarray.merge([forcings_prediction, norm_toa])
        forcings_prediction = xarray.merge([forcings_prediction, toa])
        norm_inputs_pred = norm_original_fn(inputs_pred)
        if 'total_precipitation_6hr' in norm_inputs_pred.data_vars and args.resolution == "0.25deg":
            norm_inputs_pred = norm_inputs_pred.drop_vars('total_precipitation_6hr')
        norm_static = norm_original_fn(inputs_static)
        mask = batch["mask"]
        measurements_interp = batch["weatherbench_interp"].drop_vars(["datetime", "toa_incident_solar_radiation"])
        mask = _to_jax_xarray(mask, device)["mask"] # convert from dataset to dataarray
        measurements_interp = _to_jax_xarray(measurements_interp, device)
        measurements_diff_interp = measurements_interp - inputs_pred
        norm_measurements_diff_interp = norm_diff_fn(measurements_diff_interp)

        corrected_pred_prev = corrected_pred
        corrected_pred_prognoistic, _ = repaint_fn.apply(repaint_params, state={}, rng=None, repaint_mask = mask,
                                             norm_measurements_diff_interp = norm_measurements_diff_interp,
                                             norm_inputs_pred = norm_inputs_pred,
                                             norm_forcings = norm_forcings,
                                             norm_static = norm_static,
                                             rng_batch = rng_batch,
                                             progress_bar = True)
        corrected_pred = xarray.merge([corrected_pred_prognoistic, forcings_prediction])
        if 'total_precipitation_6hr' in corrected_pred.data_vars and args.resolution == "0.25deg":
            corrected_pred = corrected_pred.drop_vars('total_precipitation_6hr')
        if corrected_pred_prev is None:
            continue
        graphcast_inputs = xarray.concat([corrected_pred_prev, corrected_pred], dim="time")
        targets_template = inputs_ground_truth.drop_vars("toa_incident_solar_radiation")
        # TODO: run 48h prediction
        inputs_pred_prev = inputs_pred
        inputs_pred, _ = graphcast_fn_jitted(graphcast_params, state={}, rng=None, 
                                          inputs=graphcast_inputs,
                                          norm_static=norm_static,
                                          norm_forcings=norm_forcings_prediction,
                                          targets_template=targets_template,)
        
        if args.dump_data:
            obs_coords = xarray.DataArray(batch['obs_coords'], dims=['batch', 'latlon', 'dim'])
            measurements_interp_output = _to_numpy_xarray(measurements_interp)
            measurements_interp_output['obs_coords'] = obs_coords
            _to_numpy_xarray(corrected_pred_prognoistic).to_zarr(f"{args.dump_data_folder}/DA_step{batch_idx}.zarr")
            measurements_interp_output.to_zarr(f"{args.dump_data_folder}/Interp_step{batch_idx}.zarr")

        diff = corrected_pred_prognoistic - inputs_ground_truth.drop_vars("toa_incident_solar_radiation")
        diff_gc = inputs_pred_prev - inputs_ground_truth.drop_vars("toa_incident_solar_radiation")
        diff_interp = measurements_interp - inputs_ground_truth.drop_vars("toa_incident_solar_radiation")

        df_da_pl, df_da_sl = calculate_stat_rmse(diff, data_type="Repaint_DA")
        df_gc_pl, df_gc_sl = calculate_stat_rmse(diff_gc, data_type="GraphCast_Pred")
        df_it_pl, df_it_sl = calculate_stat_rmse(diff_interp, data_type="Interpolation")

        df_pl = pd.concat([df_da_pl, df_gc_pl, df_it_pl])
        df_sl = pd.concat([df_da_sl, df_gc_sl, df_it_sl])

        diff_500hPa = diff.sel(level=500)
        diff_gc_500hPa = diff_gc.sel(level=500)

        val_loss = {f"val/diffusion_rmse_500hPa/{k}": jnp.sqrt(xarray_jax.unwrap_data((v*v).mean())).item() for k,v in diff_500hPa.data_vars.items()}
        val_loss_gc = {f"val/graphcast_rmse500hPa/{k}": jnp.sqrt(xarray_jax.unwrap_data((v*v).mean())).item() for k,v in diff_gc_500hPa.data_vars.items()}
        #val_loss["rank"] = args.rank

        pbar.set_postfix(val_loss_z500=val_loss[f"val/diffusion_rmse_500hPa/geopotential"], loss_gc_z500=val_loss_gc[f"val/graphcast_rmse500hPa/geopotential"])
        if args.use_wandb:
            log_dict = {**val_loss, **val_loss_gc}
            wandb.log(log_dict)
            table_pl = wandb.Table(dataframe=df_pl)
            table_sl = wandb.Table(dataframe=df_sl)
            wandb.log({"val/rmse_pressure_level": table_pl, "val/rmse_surface_level": table_sl})

def generate_repaint_fn(diffusion_checkpoint, noise_scheduler, stddev_by_level, mean_by_level, diffs_stddev_by_level):
    @hk.transform_with_state
    def repaint_fn(repaint_mask, norm_measurements_diff_interp, norm_inputs_pred, norm_forcings, norm_static, rng_batch, progress_bar=False):
        predictor = wrap_graphcast_diffusion(diffusion_checkpoint.model_config, diffusion_checkpoint.task_config, stddev_by_level, mean_by_level, diffs_stddev_by_level)
        corrected_predction = predictor.repaint_forward(repaint_mask = repaint_mask,
                                                        norm_measurements_diff_interp = norm_measurements_diff_interp,
                                                        norm_inputs_pred = norm_inputs_pred,
                                                        norm_forcings = norm_forcings,
                                                        norm_static = norm_static,
                                                        noise_scheduler = noise_scheduler,
                                                        scheduler_state = diffusion_checkpoint.scheduler_state,
                                                        num_inference_steps = args.num_repaint_inference_timesteps,
                                                        repaint_eta = args.rapaint_eta,
                                                        repaint_jump_length = args.repaint_jump_length,
                                                        repaint_jump_n_sample = args.repaint_jump_n_sample,
                                                        rng_batch = rng_batch,
                                                        progress_bar = progress_bar)
        return corrected_predction
    return repaint_fn

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument("--num_autoregressive_steps", type=int, default=10)
    parser.add_argument("--validation_year", type=int, default=2016)
    parser.add_argument("--graphcast_pred_path", type=str, default="/Data/GraphCast_sample/pred/2016_01.zarr")
    parser.add_argument("--weatherbench2_path", type=str, default="/Data/GraphCast_sample/wb2/2016_01.zarr")
    parser.add_argument("--resolution", type=str, default="1deg")
    parser.add_argument("--stats_path", type=str, default="/workspace/stats")
    parser.add_argument("--graphcast_checkpoint_path", type=str, default="/workspace/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz")
    parser.add_argument("--diffusion_6h_checkpoint_directory", type=str, default="/checkpoints_6h")
    parser.add_argument("--diffusion_48h_checkpoint_directory", type=str, default="/checkpoints_48h")
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--num_repaint_inference_timesteps", type=int, default=300)
    parser.add_argument("--rapaint_eta", type=float, default=0.0)
    parser.add_argument("--repaint_jump_length", type=int, default=10)
    parser.add_argument("--repaint_jump_n_sample", type=int, default=10)
    parser.add_argument("--mask_blur_kernel_size", type=float, default=1.5)
    parser.add_argument("--repaint_num_sparse_samples", type=int, default=4000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--fixed_measurements", action="store_true")
    parser.add_argument("--dump_data", action="store_true")
    parser.add_argument("--dump_data_folder", type=str, default="")
    parser.add_argument("--dataset_time_offset", type=int, default=0)
    parser.add_argument("--experiment_id", type=str, default="0")

    args = parser.parse_args()

    args.wandb_name += f" Nsample_{args.repaint_num_sparse_samples}"
    args.dump_data_folder += f"/Nsample_{args.repaint_num_sparse_samples}/Offset_{args.dataset_time_offset}/{args.experiment_id}"

    if args.dump_data:
        os.system(f"mkdir -p {args.dump_data_folder}")

    args.mode = "autoreg"
    if args.fixed_measurements:
        args.mode = "autoreg_fixed"
        
    if args.resolution == "1deg":
        args.downsample = True
    elif args.resolution == "0.25deg":
        args.downsample = False
        os.environ["GRAPHCAST_CHECKPOINTING"] = "True"
    else: 
        raise ValueError(f"Unknown resolution {args.resolution}")
    
    args.diffusion_6h_checkpoint_directory = os.path.join(args.diffusion_6h_checkpoint_directory, args.resolution)
    args.diffusion_48h_checkpoint_directory = os.path.join(args.diffusion_48h_checkpoint_directory, args.resolution)

    if args.use_wandb:
        wandb_key = os.environ.get("WANDB_KEY")
        wandb.login(key=wandb_key)
        name = datetime.now().strftime('%m-%d-%H:%M')
        name += "-autoreg DA" + args.wandb_name
        wandb.init(project="DA_paper", name=name, config=args)
        current_file_directory = os.path.dirname(os.path.realpath(__file__))
        wandb.run.log_code(current_file_directory)

    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_normalization(args.stats_path)
    graphcast_model_config, graphcast_task_config, graphcast_params = load_model_checkpoint(args.graphcast_checkpoint_path)

    diffusion_6h_checkpoint = load_diffusion_checkpoint(Path(args.diffusion_6h_checkpoint_directory))
    diffusion_48h_checkpoint = load_diffusion_checkpoint(Path(args.diffusion_48h_checkpoint_directory))
    args.num_train_timesteps_6h = diffusion_6h_checkpoint.num_train_timesteps
    args.num_train_timesteps_48h = diffusion_48h_checkpoint.num_train_timesteps
    noise_scheduler_6h = FlaxDDPMScheduler(args.num_train_timesteps_6h,
                                        beta_schedule=args.ddpm_beta_schedule,
                                        prediction_type="epsilon")
    noise_scheduler_48h= FlaxDDPMScheduler(args.num_train_timesteps_48h,
                                        beta_schedule=args.ddpm_beta_schedule,
                                        prediction_type="epsilon")

    repaint_6h_fn = generate_repaint_fn(diffusion_6h_checkpoint, noise_scheduler_6h, stddev_by_level, mean_by_level, diffs_stddev_by_level)
    repaint_48h_fn = generate_repaint_fn(diffusion_48h_checkpoint, noise_scheduler_48h, stddev_by_level, mean_by_level, diffs_stddev_by_level)
    repaint_6h_params = diffusion_6h_checkpoint.params
    repaint_48h_params = diffusion_48h_checkpoint.params


    
    device = jax.local_devices()[0]
    rng = jax.random.PRNGKey(args.random_seed)
    

    
    @hk.transform_with_state
    def graphcast_fn(inputs: xarray.Dataset,
                     norm_forcings: xarray.Dataset,
                     norm_static: xarray.Dataset,
                     targets_template: xarray.Dataset):
        predictor = wrap_graphcast_prediction(graphcast_model_config,
                                              graphcast_task_config,
                                              diffs_stddev_by_level,
                                              mean_by_level,
                                              stddev_by_level,
                                              normalize=True)
        return predictor.predict(inputs=inputs,
                                 norm_forcings=norm_forcings,
                                 norm_static=norm_static,
                                 targets_template=targets_template)
    
    norm_diff_fn = functools.partial(normalization.normalize, scales=diffs_stddev_by_level, locations=None)
    norm_original_fn = functools.partial(normalization.normalize, scales=stddev_by_level, locations=mean_by_level)

    validate_dataset = GraphCastDiffusionDataset(args.graphcast_pred_path.format(args.validation_year), 
                                                 args.weatherbench2_path.format(args.validation_year),
                                                 sample_slice=slice(args.dataset_time_offset, args.dataset_time_offset + args.num_autoregressive_steps),
                                                 downsample=args.downsample,
                                                 num_sparse_samples=args.repaint_num_sparse_samples,
                                                 blur_kernel_size=args.mask_blur_kernel_size,
                                                 fixed_measurements=args.fixed_measurements,)
    


    autoregressive_assimilation(graphcast_fn, repaint_6h_fn, repaint_48h_fn, norm_original_fn, norm_diff_fn, graphcast_params, repaint_6h_params, repaint_48h_params, validate_dataset, args, device, rng)