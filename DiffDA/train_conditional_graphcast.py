import math
from pathlib import Path

import haiku as hk
import optax
import jax
import jax.numpy as jnp
import xarray
import checkpoint
from graphcast import xarray_jax
from graphcast import xarray_tree
from graphcast import normalization
from diffusers import FlaxDDPMScheduler
from diffusers.schedulers.scheduling_utils_flax import add_noise_common
import functools
import numpy as np

from diffusion_common import get_forcing, wrap_graphcast, _to_jax_xarray, validation_step

from dataloader import GraphCastDiffusionDataset, load_normalization, load_model_checkpoint, DiffusionDataLoader
import jax_dataloader as jdl
from argparse import ArgumentParser
from typing import List, Tuple, NamedTuple, Callable, Sequence, Mapping, Union
from tqdm import tqdm
from time import time
import nvtx
import wandb
from datetime import datetime
import resource
import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

#https://github.com/google-deepmind/dm-haiku/blob/main/examples/mnist.py
class TrainingState(NamedTuple):
  params: hk.Params
  #avg_params: hk.Params
  opt_state: optax.OptState



def mem_snapshot():
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    return gb

def clear_caches():
  """Clear all compilation and staging caches."""
  from jax._src import linear_util, util, pjit, dispatch
  from jax._src.lib import xla_client
  from jax._src.api import _pmap_cache_clears
  # Clear all lu.cache and util.weakref_lru_cache instances (used for staging
  # and Python-dispatch compiled executable caches).
  linear_util.clear_all_caches()
  #util.clear_all_weakref_lru_caches() # <- this will cause 139 error!

  # Clear all C++ compiled executable caches for pjit
  pjit._cpp_pjit_cache.clear()
  xla_client._xla.PjitFunctionCache.clear_all()

  # Clear all C++ compiled executable caches for pmap
  for fun in _pmap_cache_clears:
    fun._cache_clear()

  # Clear particular util.cache instances.
  dispatch.xla_primitive_callable.cache_clear()

def reset_device_memory():
    import jax.lib.xla_bridge
    backend = jax.lib.xla_bridge.get_backend()
    for buf in backend.live_buffers(): 
        buf.delete()


def add_noise(rng, inputs, timesteps, scheduler_state):
    def noise_fn(da):
        rng_i = jax.random.fold_in(rng, hash(da.name))
        noise = jax.random.normal(rng_i, da.shape, dtype=da.dtype)
        return xarray.zeros_like(da) + noise
    def add_fn(da, noise):
        return add_noise_common(scheduler_state, da, noise, timesteps)
    
    ds_noise = xarray_tree.map_structure(noise_fn, inputs)
    ds_inputs_noise = xarray_tree.map_structure(add_fn, inputs, ds_noise)
    return ds_inputs_noise, ds_noise

_grad_and_update_jitted = None

def update(loss_fn: hk.TransformedWithState, optimizer, training_state: TrainingState, state, norm_inputs_noise, norm_inputs_pred, targets_noise, norm_forcings, norm_static, rng):
    def _aux(params, state, norm_inputs_noise, norm_inputs_pred, targets_noise, norm_forcings, norm_static):
        (loss, diagnostics), next_state = loss_fn.apply(params, state, rng, norm_inputs_noise, norm_inputs_pred, targets_noise, norm_forcings, norm_static)
        return loss, (diagnostics, next_state)
    
    def _grad_and_update(training_state, state, norm_inputs_noise, norm_inputs_pred, targets_noise, norm_forcings, norm_static):
        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(training_state.params,
                                                                                        state,
                                                                                        norm_inputs_noise,
                                                                                        norm_inputs_pred,
                                                                                        targets_noise,
                                                                                        norm_forcings,
                                                                                        norm_static)
        if args.distributed:
            grads = jax.lax.pmean(grads, axis_name=pmap_dim)
        updates, opt_state = optimizer.update(grads, training_state.opt_state, training_state.params)
        params = optax.apply_updates(training_state.params, updates)
        return loss, diagnostics, next_state, TrainingState(params, opt_state)

    # TODO add reduce_axes=('batch',)
    global _grad_and_update_jitted
    if args.distributed:
        pmap_dim = "device_count"
        if _grad_and_update_jitted is None:
            _grad_and_update_jitted = xarray_jax.pmap(_grad_and_update, dim=pmap_dim, axis_name=pmap_dim)
        norm_inputs_noise = norm_inputs_noise.expand_dims(pmap_dim, axis=0)
        norm_inputs_pred = norm_inputs_pred.expand_dims(pmap_dim, axis=0)
        targets_noise = targets_noise.expand_dims(pmap_dim, axis=0)
        norm_forcings = norm_forcings.expand_dims(pmap_dim, axis=0)
        norm_static = norm_static.expand_dims(pmap_dim, axis=0)
        training_state = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), training_state)
        state = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
        loss, diagnostics, next_state, training_state = _grad_and_update_jitted(training_state, state, norm_inputs_noise, norm_inputs_pred, targets_noise, norm_forcings, norm_static)
        loss = jnp.squeeze(loss, axis=0)
        diagnostics = jax.tree_map(lambda x: jnp.squeeze(x, axis=0), diagnostics)
        next_state = jax.tree_map(lambda x: jnp.squeeze(x, axis=0), next_state)
        training_state = jax.tree_map(lambda x: jnp.squeeze(x, axis=0), training_state)
    else:
        if _grad_and_update_jitted is None:
            _grad_and_update_jitted = jax.jit(_grad_and_update, donate_argnames=["training_state", "state"])
        loss, diagnostics, next_state, training_state = _grad_and_update_jitted(training_state, state, norm_inputs_noise, norm_inputs_pred, targets_noise, norm_forcings, norm_static)
    return loss, diagnostics, next_state, training_state


def train_model(loss_fn: hk.TransformedWithState,
                forward_ddpm_fn: hk.TransformedWithState,
                forward_repaint_fn: hk.TransformedWithState,
                norm_diff_fn: Callable,
                norm_original_fn: Callable,
                checkpoint: checkpoint.TrainingCheckpoint,
                args,
                epochs: int=5):

    rng_original = checkpoint.rng

    def preprocess(inputs_ground_truth, inputs_pred, norm_forcings, inputs_static, rng_noise, timesteps):
        norm_soa = norm_original_fn(inputs_ground_truth["toa_incident_solar_radiation"])
        inputs_ground_truth = inputs_ground_truth.drop_vars("toa_incident_solar_radiation")
        norm_forcings = xarray.merge([norm_forcings, norm_soa])
        train_inputs_diff = inputs_ground_truth - inputs_pred
        norm_inputs_diff = norm_diff_fn(train_inputs_diff)
        norm_inputs_pred = norm_original_fn(inputs_pred)
        norm_static = norm_original_fn(inputs_static)

        norm_inputs_noise, targets_noise = add_noise_fn(rng_noise, norm_inputs_diff, timesteps)
        return norm_inputs_noise, norm_inputs_pred, targets_noise, norm_forcings, norm_static
    
    if args.rank == 0:
        print(f"Using device {args.device}")
    jax.config.update('jax_platform_name', args.device)
    device = jax.local_devices(backend=args.device)[0]

    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=1e-5,
                                                peak_value=args.learning_rate,
                                                warmup_steps=(1/6)*args.num_train_steps_per_epoch*args.total_epochs,
                                                decay_steps=(5/6)*args.num_train_steps_per_epoch*args.total_epochs,
                                                end_value=3e-6)
    @optax.inject_hyperparams
    def get_optimizer(learning_rate):
        # TODO: add gradient accumulation, and tune weight decay parameters
        return optax.chain(
            optax.clip(0.5),
            optax.adamw(learning_rate=learning_rate),
            )
    optimizer = get_optimizer(lr_schedule)
    update_fn = functools.partial(update, optimizer=optimizer)
    add_noise_fn= functools.partial(add_noise, scheduler_state=checkpoint.scheduler_state.common)
    preprocess_jitted = jax.jit(preprocess, donate_argnames=["inputs_ground_truth", "inputs_pred", "norm_forcings", "inputs_static"])
    state = {}
    if checkpoint.opt_state is None:
        initial_opt_state = optimizer.init(checkpoint.params)
    else:
        initial_opt_state = checkpoint.opt_state
    training_state = TrainingState(checkpoint.params, initial_opt_state)

    iter_count = 0

    shared_rng, rng = jax.random.split(rng_original, 2)
    if args.distributed:
        rng = jax.random.fold_in(rng, args.rank)

    # Continue the epoch where we left off
    for epoch in range(checkpoint.epoch + 1, checkpoint.epoch + epochs + 1):

        if epoch > args.total_epochs:
            print(f"Warning: epoch {epoch} exceeds total epochs {args.total_epochs}")

        #global _grad_and_update_jitted
        #_grad_and_update_jitted = None
        #reset_device_memory()
        #clear_caches() # avoid OOM after several epochs

        rng_epoch = jax.random.fold_in(rng, epoch)
        rng_epoch_training, rng_epoch_validation_ddpm, rng_epoch_validation_repaint = jax.random.split(rng_epoch, 3)

        if args.rank == 0 and (epoch % args.validation_every_n_epoch == 0 or epoch == checkpoint.epoch + epochs):
            validate_dataset = GraphCastDiffusionDataset(args.graphcast_pred_path.format(args.validation_year), 
                                                         args.weatherbench2_path.format(args.validation_year),
                                                         sample_slice=slice(0, args.num_validation_samples_per_year),
                                                         downsample=args.downsample,
                                                         num_sparse_samples=args.repaint_num_sparse_samples,
                                                         blur_kernel_size=args.mask_blur_kernel_size,
                                                         offset=args.graphcast_pred_offset,
                                                         disable_pred_offset_check=args.disable_pred_offset_check)
            validate_dataloader = jdl.DataLoader(validate_dataset, backend="jax", batch_size=args.validation_batch_size, shuffle=False, drop_last=True)
            # Validation Repaint pipeline
            if not args.disable_validation_repaint:
                validation_step(forward_repaint_fn, norm_original_fn, norm_diff_fn, training_state.params, state, validate_dataloader, args, device, rng_epoch_validation_repaint, mode="repaint")
            # Validation DDPM pipeline
            if not args.disable_validation_ddpm:
                validation_step(forward_ddpm_fn, norm_original_fn, norm_diff_fn, training_state.params, state, validate_dataloader, args, device, rng_epoch_validation_ddpm, mode="ddpm")
            

        #if args.distributed:
        #    args.comm.Barrier()

        if args.disable_training:
            exit(0)

        # Compute subset of years to train on
        rng_years_permutation = jax.random.fold_in(shared_rng, epoch)
        samples = _samples_for_rank(args.years_start,
                                  args.years_end,
                                  args.rank,
                                  args.total_rank,
                                  rng_years_permutation,
                                  True)
        print("Rank:", args.rank, " samples: ", samples.items())

        dataloader = DiffusionDataLoader(args.graphcast_pred_path,
                                         args.weatherbench2_path,
                                         samples_for_year = samples,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         downsample=args.downsample,
                                         offset=args.graphcast_pred_offset,
                                         disable_pred_offset_check=args.disable_pred_offset_check,)
        train_iter = iter(dataloader)
        pbar = range(len(dataloader))
        if args.rank == 0:
            pbar = tqdm(pbar, desc=f"Epoch {epoch}")
        for batch_idx in pbar:
            if args.profile and args.profile_kind == "tensorboard" and iter_count == 1:
                    jax.profiler.start_trace(args.profile_path, create_perfetto_trace=True)
            with jax.profiler.TraceAnnotation(f"step {iter_count}") as jaxprof, nvtx.annotate(f"step {iter_count}") as nvtxprof:
                t1 = time()
                rng_batch = jax.random.fold_in(rng_epoch_training, batch_idx)
                rng_timestep, rng_noise = jax.random.split(rng_batch, num=2)
                timesteps = np.asarray(jax.random.randint(rng_timestep, shape=(args.batch_size,), minval=0, maxval=args.num_train_timesteps))
                batch = next(train_iter)
                inputs_pred = batch['graphcast']
                inputs_ground_truth = batch['weatherbench']
                inputs_static = batch['static']
                datetime = inputs_ground_truth.datetime
                lon = inputs_ground_truth.lon
                norm_forcings = get_forcing(datetime, lon, timesteps, args.num_train_timesteps, batch_size=args.batch_size)
                norm_forcings = _to_jax_xarray(norm_forcings, device)
                inputs_pred = _to_jax_xarray(inputs_pred.drop_vars("datetime"), device)
                inputs_ground_truth = _to_jax_xarray(inputs_ground_truth.drop_vars("datetime"), device)
                inputs_static = _to_jax_xarray(inputs_static, device)
                timesteps = jnp.asarray(timesteps)

                norm_inputs_noise, norm_inputs_pred, targets_noise, norm_forcings, norm_static = preprocess_jitted(inputs_ground_truth, inputs_pred, norm_forcings, inputs_static, rng_noise, timesteps)

                # for 0.25deg model, we use the graphcast_operational model which do not take in tp as an input
                if 'total_precipitation_6hr' in norm_inputs_pred.data_vars and args.resolution == "0.25deg":
                    norm_inputs_pred = norm_inputs_pred.drop_vars('total_precipitation_6hr')
                    norm_inputs_noise = norm_inputs_noise.drop_vars('total_precipitation_6hr')

                assert len(norm_inputs_noise.time) == 1 and len(norm_inputs_pred.time) == 1
                t2 = time()
                loss, diagnostics, state, training_state = update_fn(
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    training_state=training_state,
                    state=state,
                    norm_inputs_noise=norm_inputs_noise,
                    norm_inputs_pred=norm_inputs_pred,
                    targets_noise=targets_noise,
                    norm_forcings=norm_forcings,
                    norm_static=norm_static,
                    rng=rng_batch,)
                #jax.block_until_ready(grads)
                if args.profile:
                    jax.block_until_ready(loss)
                t3 = time()
                # Only log occasionally to reduce overhead
                if args.rank == 0 and iter_count % args.log_interval == 0:
                    learning_rate = training_state.opt_state.hyperparams["learning_rate"]
                    pbar.set_postfix(loss=loss, preproc_time=t2-t1, grad_time=t3-t2, lr=learning_rate)
                    if args.use_wandb:
                        wandb.log({"train_loss": loss, "preproc_time": t2-t1, "grad_time": t3-t2, "learning_rate": learning_rate})
            iter_count += 1
            if args.debug and iter_count > 10:
                break
            if args.profile:
                if iter_count == 1:
                    print(f"Iter 0 time: {t3-t1} s", flush=True)
                if iter_count > 4:
                    if args.profile_kind == "tensorboard":
                        jax.profiler.stop_trace()
                    return

        
        if args.rank == 0:
            # Save checkpoint
            checkpoint.save_checkpoint(
                Path(args.checkpoint_directory),
                checkpoint.TrainingCheckpoint(
                    params=training_state.params,
                    opt_state=training_state.opt_state,
                    task_config=checkpoint.task_config,
                    model_config=checkpoint.model_config,
                    scheduler_state=checkpoint.scheduler_state,
                    epoch=epoch,
                    rng=rng_original,
                    num_train_timesteps=args.num_train_timesteps,
            ))

def _samples_for_rank(years_start: int,
                      years_end: int,
                      rank: int,
                      num_ranks: int,
                      shared_key: jax.Array,
                      shuffle: bool = True) -> Mapping[int, Union[slice, None]]:
    """
    Returns the samples that a given rank should train on in the current epoch.
    Note: might not use all years in each epoch in order to ensure each rank gets the same number of years.
    If the number of ranks exceeds the number of years, each year is split into equal slices and each rank trains
    on one slice of a year.
    Assumes a lead time of 48h and 6h time resolution.
    @param years_start: Start of year range
    @param years_end: End of year range
    @param rank: Rank of current process
    @param num_ranks: Number of ranks overall
    @param shared_key: PRNG key SHARED by all ranks
    @param shuffle: If true, shuffle the number of years. When shuffle is true, all years will be sampled eventually
    and are equally likely to occur in any epoch.
    @return: Mapping from year to slices of iterations.
    """

    all_years = jax.numpy.arange(years_start, years_end)
    if shuffle:
        all_years = jax.random.permutation(key=shared_key, x=all_years)

    n_years = years_end-years_start
    if num_ranks <= n_years:
        rank_nr_years = n_years // num_ranks  # Size of each part
        first_year = rank * rank_nr_years
        years = all_years[first_year: first_year + rank_nr_years]

        return {y: None for y in years.tolist()}
    else:
        number_of_ranks_per_year = int(math.ceil(num_ranks / n_years))
        rank_year = all_years[rank // number_of_ranks_per_year].item()

        index_within_year = rank % number_of_ranks_per_year

        # TODO Generalize for different lead times
        size_of_slize = (365 * 4 - 9) // number_of_ranks_per_year

        start = index_within_year * size_of_slize
        stop = start + size_of_slize
        return {rank_year: slice(start, stop)}

def main(args):
    if args.distributed:
        #from mpi4py import MPI
        #comm = MPI.COMM_WORLD
        jax.distributed.initialize()
        assert jax.local_device_count() == 1, "Only support single GPU per rank"
        args.rank = jax.process_index()
        args.total_rank = jax.process_count()
        #args.comm = comm
        #assert args.total_rank == comm.Get_size(), f"Total rank does not match, {args.total_rank}, {comm.Get_size()}"
        device = jax.local_devices()[0]
        print(f"Rank {args.rank + 1}/{args.total_rank} initialized, device: {device}")
    args.num_train_steps_per_epoch = args.num_train_steps_per_epoch // args.total_rank

    if args.use_wandb and args.rank == 0:
        wandb_key = os.environ.get("WANDB_KEY")
        wandb.login(key=wandb_key)
        name = datetime.now().strftime('%m-%d-%H:%M')
        if args.disable_training:
            name += "-validation"
        wandb.init(project="Diffusion GraphCast", name=name, config=args)
        current_file_directory = os.path.dirname(os.path.realpath(__file__))
        wandb.run.log_code(current_file_directory)
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_normalization(args.stats_path)

    norm_diff_fn = functools.partial(normalization.normalize, scales=diffs_stddev_by_level, locations=None)
    norm_original_fn = functools.partial(normalization.normalize, scales=stddev_by_level, locations=mean_by_level)

    # Load the latest checkpoint from the directory
    checkpoint = checkpoint.load_checkpoint(Path(args.checkpoint_directory))
    noise_scheduler = FlaxDDPMScheduler(args.num_train_timesteps,
                                        beta_schedule=args.ddpm_beta_schedule,
                                        prediction_type="epsilon")
    # If none yet, start training from pretrained graphcast model
    if checkpoint is None:
        model_config, task_config, params = load_model_checkpoint(args.checkpoint_path)

        scheduler_state = noise_scheduler.create_state()
        checkpoint = checkpoint.TrainingCheckpoint(
            model_config=model_config,
            task_config=task_config,
            params=params,
            opt_state=None,
            scheduler_state=scheduler_state,
            rng=jax.random.PRNGKey(args.random_seed),
            epoch=-1,
            num_train_timesteps=args.num_train_timesteps,
        )
    if args.ignore_optstate_from_checkpoint:
        from dataclasses import replace
        checkpoint = replace(checkpoint, opt_state=None)

    if checkpoint.num_train_timesteps != args.num_train_timesteps:
        raise ValueError(f"Number of training timesteps in checkpoint ({checkpoint.num_train_timesteps}) does not match number of training timesteps in args ({args.num_train_timesteps})")

    jdl_config = jdl.get_config()
    jdl_key = jax.random.fold_in(checkpoint.rng, checkpoint.epoch)
    jdl_key = jax.random.fold_in(jdl_key, args.rank)
    jdl_config.global_seed = jax.random.randint(jdl_key,
                                                [1],
                                                minval=-10e6,
                                                maxval=10e6)[0]

    @hk.transform_with_state
    def loss_fn(norm_inputs_noise, norm_inputs_pred, targets_noise, norm_forcings, norm_static):
        predictor = wrap_graphcast(checkpoint.model_config, checkpoint.task_config, stddev_by_level, mean_by_level, diffs_stddev_by_level)
        loss, diagnostics = predictor.loss(norm_inputs_noise, norm_inputs_pred, targets_noise, norm_forcings, norm_static)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics))

    @hk.transform_with_state
    def forward_ddpm_fn(norm_inputs_pred, norm_forcings, norm_static, rng_batch, progress_bar=False):
        predictor = wrap_graphcast(checkpoint.model_config, checkpoint.task_config, stddev_by_level, mean_by_level, diffs_stddev_by_level)
        corrected_predction = predictor(norm_inputs_pred, norm_forcings, norm_static, noise_scheduler, checkpoint.scheduler_state, args.num_inference_timesteps, rng_batch, progress_bar=progress_bar)
        return corrected_predction
    
    @hk.transform_with_state
    def forward_repaint_fn(mask, norm_measurements_diff_interp, norm_inputs_pred, norm_forcings, norm_static, rng_batch, progress_bar=False):
        predictor = wrap_graphcast(checkpoint.model_config, checkpoint.task_config, stddev_by_level, mean_by_level, diffs_stddev_by_level)
        corrected_predction = predictor.repaint_forward(repaint_mask = mask,
                                                        norm_measurements_diff_interp = norm_measurements_diff_interp,
                                                        norm_inputs_pred = norm_inputs_pred,
                                                        norm_forcings = norm_forcings,
                                                        norm_static = norm_static,
                                                        noise_scheduler = noise_scheduler,
                                                        scheduler_state = checkpoint.scheduler_state,
                                                        num_inference_steps = args.repaint_num_inference_timesteps,
                                                        repaint_eta = args.rapaint_eta,
                                                        repaint_jump_length = args.repaint_jump_length,
                                                        repaint_jump_n_sample = args.repaint_jump_n_sample,
                                                        rng_batch = rng_batch,
                                                        progress_bar = progress_bar)
        return corrected_predction

    train_model(loss_fn, forward_ddpm_fn, forward_repaint_fn, norm_diff_fn, norm_original_fn, checkpoint, args, args.epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--graphcast_pred_path", type=str, default="/Data/GraphCast_sample/pred/2016_01.zarr")
    parser.add_argument("--weatherbench2_path", type=str, default="/Data/GraphCast_sample/wb2/2016_01.zarr")
    parser.add_argument("--graphcast_pred_offset", type=int, default=9) # 9 for 48h lead time, 2 step prediction
    parser.add_argument("--disable_pred_offset_check", action="store_true")
    parser.add_argument("--stats_path", type=str, default="/workspace/stats")
    parser.add_argument("--profile_path", type=str, default="/workspace/profile")
    parser.add_argument("--profile_kind", type=str, default="tensorboard")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--validation_batch_size", type=int, default=1)
    parser.add_argument("--validation_year", type=int, default=2016)
    parser.add_argument("--validation_every_n_epoch", type=int, default=1)
    parser.add_argument("--num_samples_per_year", type=int, default=None)
    parser.add_argument("--num_validation_samples_per_year", type=int, default=1)
    parser.add_argument("--resolution", type=str, default="1deg")
    parser.add_argument("--checkpoint_path", type=str, default="/workspace/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz")
    parser.add_argument("--checkpoint_directory", type=str, default="/checkpoints")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_timesteps", type=int, default=1000)
    parser.add_argument("--num_train_samples_per_epoch", type=int, default=52560) # (total samples)/(batch_size)
    parser.add_argument("--mask_blur_kernel_size", type=float, default=1.5)
    parser.add_argument("--repaint_num_inference_timesteps", type=int, default=300)
    parser.add_argument("--rapaint_eta", type=float, default=0.0)
    parser.add_argument("--repaint_jump_length", type=int, default=10)
    parser.add_argument("--repaint_jump_n_sample", type=int, default=10)
    parser.add_argument("--repaint_num_sparse_samples", type=int, default=1000)
    parser.add_argument("--disable_validation_ddpm", action="store_true")
    parser.add_argument("--disable_validation_repaint", action="store_true")
    parser.add_argument("--disable_training", action="store_true")
    parser.add_argument("--ignore_optstate_from_checkpoint", action="store_true")
    parser.add_argument("--years_start", type=int, default=1979)
    parser.add_argument("--years_end", type=int, default=2016)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--total_epochs", type=int, default=-1)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")

    args = parser.parse_args()
    args.num_train_steps_per_epoch = args.num_train_samples_per_epoch // args.batch_size
    args.rank = 0
    args.total_rank = 1
    if args.total_epochs == -1:
        args.total_epochs = args.epochs

    if args.resolution == "1deg":
        args.downsample = True
    elif args.resolution == "0.25deg":
        args.downsample = False
        os.environ["GRAPHCAST_CHECKPOINTING"] = "True"
    else:
        raise ValueError(f"Unknown resolution {args.resolution}")
    
    args.checkpoint_directory = os.path.join(args.checkpoint_directory, args.resolution)

    main(args)

