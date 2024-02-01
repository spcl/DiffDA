# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrappers for Predictors which allow them to work with normalized data.

The Predictor which is wrapped sees normalized inputs and targets, and makes
normalized predictions. The wrapper handles translating the predictions back
to the original domain.
"""

import logging
from typing import Optional, Tuple, Callable

from graphcast import predictor_base
from graphcast import xarray_tree, xarray_jax
from graphcast import losses
import xarray
from tqdm import tqdm

def normalize(values: xarray.Dataset,
              scales: xarray.Dataset,
              locations: Optional[xarray.Dataset],
              ) -> xarray.Dataset:
  """Normalize variables using the given scales and (optionally) locations."""
  def normalize_array(array):
    if array.name is None:
      raise ValueError(
          "Can't look up normalization constants because array has no name.")
    if locations is not None:
      if array.name in locations:
        array = array - locations[array.name].astype(array.dtype)
      else:
        logging.warning('No normalization location found for %s', array.name)
    if array.name in scales:
      array = array / scales[array.name].astype(array.dtype)
    else:
      logging.warning('No normalization scale found for %s', array.name)
    return array
  return xarray_tree.map_structure(normalize_array, values)


def unnormalize(values: xarray.Dataset,
                scales: xarray.Dataset,
                locations: Optional[xarray.Dataset],
                ) -> xarray.Dataset:
  """Unnormalize variables using the given scales and (optionally) locations."""
  def unnormalize_array(array):
    if array.name is None:
      raise ValueError(
          "Can't look up normalization constants because array has no name.")
    if array.name in scales:
      array = array * scales[array.name].astype(array.dtype)
    else:
      logging.warning('No normalization scale found for %s', array.name)
    if locations is not None:
      if array.name in locations:
        array = array + locations[array.name].astype(array.dtype)
      else:
        logging.warning('No normalization location found for %s', array.name)
    return array
  return xarray_tree.map_structure(unnormalize_array, values)


class InputsAndResiduals(predictor_base.Predictor):
  """Wraps with a residual connection, normalizing inputs and target residuals.

  The inner predictor is given inputs that are normalized using `locations`
  and `scales` to roughly zero-mean unit variance.

  For target variables that are present in the inputs, the inner predictor is
  trained to predict residuals (target - last_frame_of_input) that have been
  normalized using `residual_scales` (and optionally `residual_locations`) to
  roughly unit variance / zero mean.

  This replaces `residual.Predictor` in the case where you want normalization
  that's based on the scales of the residuals.

  Since we return the underlying predictor's loss on the normalized residuals,
  if the underlying predictor is a sum of per-variable losses, the normalization
  will affect the relative weighting of the per-variable loss terms (hopefully
  in a good way).

  For target variables *not* present in the inputs, the inner predictor is
  trained to predict targets directly, that have been normalized in the same
  way as the inputs.

  The transforms applied to the targets (the residual connection and the
  normalization) are applied in reverse to the predictions before returning
  them.
  """

  def __init__(
      self,
      predictor: predictor_base.Predictor,
      stddev_by_level: xarray.Dataset,
      mean_by_level: xarray.Dataset,
      diffs_stddev_by_level: xarray.Dataset):
    self._predictor = predictor
    self._scales = stddev_by_level
    self._locations = mean_by_level
    self._residual_scales = diffs_stddev_by_level
    self._residual_locations = None

  def _unnormalize_prediction_and_add_input(self, inputs, norm_prediction):
    if norm_prediction.sizes.get('time') != 1:
      raise ValueError(
          'normalization.InputsAndResiduals only supports predicting a '
          'single timestep.')
    if norm_prediction.name in inputs:
      # Residuals are assumed to be predicted as normalized (unit variance),
      # but the scale and location they need mapping to is that of the residuals
      # not of the values themselves.
      prediction = unnormalize(
          norm_prediction, self._residual_scales, self._residual_locations)
      # A prediction for which we have a corresponding input -- we are
      # predicting the residual:
      last_input = inputs[norm_prediction.name].isel(time=-1)
      prediction += last_input
      return prediction
    else:
      # A predicted variable which is not an input variable. We are predicting
      # it directly, so unnormalize it directly to the target scale/location:
      return unnormalize(norm_prediction, self._scales, self._locations)

  def _subtract_input_and_normalize_target(self, inputs, target):
    if target.sizes.get('time') != 1:
      raise ValueError(
          'normalization.InputsAndResiduals only supports wrapping predictors'
          'that predict a single timestep.')
    if target.name in inputs:
      target_residual = target
      last_input = inputs[target.name].isel(time=-1)
      target_residual -= last_input
      return normalize(
          target_residual, self._residual_scales, self._residual_locations)
    else:
      return normalize(target, self._scales, self._locations)

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs
               ) -> xarray.Dataset:
    norm_inputs = normalize(inputs, self._scales, self._locations)
    norm_forcings = normalize(forcings, self._scales, self._locations)
    norm_predictions = self._predictor(
        norm_inputs, targets_template, forcings=norm_forcings, **kwargs)
    return xarray_tree.map_structure(
        lambda pred: self._unnormalize_prediction_and_add_input(inputs, pred),
        norm_predictions)
  
  def predict(self,
              inputs: xarray.Dataset,
              norm_forcings: xarray.Dataset,
              norm_static: xarray.Dataset,
              targets_template: xarray.Dataset,
              **kwargs
              ) -> xarray.Dataset:
    norm_inputs = normalize(inputs, self._scales, self._locations)
    norm_inputs = xarray.merge([norm_inputs, norm_static])
    norm_predictions = self._predictor(
        norm_inputs, targets_template, forcings=norm_forcings, **kwargs)
    return xarray_tree.map_structure(
        lambda pred: self._unnormalize_prediction_and_add_input(inputs, pred),
        norm_predictions)

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: xarray.Dataset,
           **kwargs,
           ) -> predictor_base.LossAndDiagnostics:
    """Returns the loss computed on normalized inputs and targets."""
    norm_inputs = normalize(inputs, self._scales, self._locations)
    norm_forcings = normalize(forcings, self._scales, self._locations)
    norm_target_residuals = xarray_tree.map_structure(
        lambda t: self._subtract_input_and_normalize_target(inputs, t),
        targets)
    return self._predictor.loss(
        norm_inputs, norm_target_residuals, forcings=norm_forcings, **kwargs)
  
  def loss_normalized_target(self,
           inputs: xarray.Dataset,
           norm_targets: xarray.Dataset,
           forcings: xarray.Dataset,
           **kwargs,
           ) -> predictor_base.LossAndDiagnostics:
    """Returns the loss computed on normalized inputs and (already) normalized targets."""
    norm_inputs = normalize(inputs, self._scales, self._locations)
    norm_forcings = normalize(forcings, self._scales, self._locations)
    return self._predictor.loss(
        norm_inputs, norm_targets, forcings=norm_forcings, **kwargs)

  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      **kwargs,
      ) -> Tuple[predictor_base.LossAndDiagnostics,
                 xarray.Dataset]:
    """The loss computed on normalized data, with unnormalized predictions."""
    norm_inputs = normalize(inputs, self._scales, self._locations)
    norm_forcings = normalize(forcings, self._scales, self._locations)
    norm_target_residuals = xarray_tree.map_structure(
        lambda t: self._subtract_input_and_normalize_target(inputs, t),
        targets)
    (loss, scalars), norm_predictions = self._predictor.loss_and_predictions(
        norm_inputs, norm_target_residuals, forcings=norm_forcings, **kwargs)
    predictions = xarray_tree.map_structure(
        lambda pred: self._unnormalize_prediction_and_add_input(inputs, pred),
        norm_predictions)
    return (loss, scalars), predictions

import diffusers
import jax
import jax.numpy as jnp
import functools

def add_noise(da, std_dev, rng):
  rng_var_ti = jax.random.fold_in(rng, hash(da.name))
  noise = jax.random.normal(rng_var_ti, shape=da.shape, dtype=da.dtype)
  return std_dev * noise + da

def has_nan(ds):
  has_nan_list = []
  for v in ds.data_vars:
    has_nan_v = jnp.isnan(xarray_jax.unwrap_data(ds[v])).any()
    if has_nan_v:
      has_nan_list.append(v)
  return has_nan_list

class InputsAndResidualsForDiffusion(predictor_base.Predictor):
  def __init__(
      self,
      predictor: predictor_base.Predictor,
      stddev_by_level: xarray.Dataset,
      mean_by_level: xarray.Dataset,
      diffs_stddev_by_level: xarray.Dataset):
    self._predictor = predictor
    self._scales = stddev_by_level
    self._locations = mean_by_level
    self._residual_scales = diffs_stddev_by_level
    self._residual_locations = None


  def _unnormalize_prediction_and_add_input(self, inputs, norm_prediction):
    if norm_prediction.sizes.get('time') != 1:
      raise ValueError(
          'normalization.InputsAndResiduals only supports predicting a '
          'single timestep.')
    if norm_prediction.name in inputs:
      # Residuals are assumed to be predicted as normalized (unit variance),
      # but the scale and location they need mapping to is that of the residuals
      # not of the values themselves.
      prediction = unnormalize(
          norm_prediction, self._residual_scales, self._residual_locations)
      # A prediction for which we have a corresponding input -- we are
      # predicting the residual:
      last_input = inputs[norm_prediction.name].isel(time=-1)
      prediction += last_input
      return prediction
    else:
      # A predicted variable which is not an input variable. We are predicting
      # it directly, so unnormalize it directly to the target scale/location:
      return unnormalize(norm_prediction, self._scales, self._locations)
    
  def _get_model_output(self,
            timestep: int,
            num_train_timesteps: int,
            norm_inputs_pred: xarray.Dataset,
            norm_inputs_noise: xarray.Dataset,
            norm_forcings: xarray.Dataset,
            norm_static: xarray.Dataset,
            **kwargs):
    timestep_embedding_template = xarray.zeros_like(norm_forcings["year_progress_sin"])
    timestep_embedding = timestep / num_train_timesteps * 2 - 1
    norm_forcings = norm_forcings.assign(year_progress_sin=timestep_embedding_template + timestep_embedding)
    
    norm_inputs = xarray.concat([norm_inputs_noise, norm_inputs_pred], dim='time')
    norm_inputs = xarray.merge([norm_inputs, xarray.concat([norm_forcings, norm_forcings], dim='time'), norm_static])
    targets_template = xarray.zeros_like(norm_inputs_pred)
    if "total_precipitation_6hr" not in targets_template.data_vars:
      targets_template = targets_template.assign(total_precipitation_6hr=xarray.zeros_like(norm_inputs_pred["mean_sea_level_pressure"]))
    
    model_output = self._predictor(
        norm_inputs, targets_template, forcings=norm_forcings, **kwargs)
    return model_output
  
  def _compute_x0(self, timestep, num_train_timesteps, norm_inputs_pred, norm_inputs_noise, norm_forcings, norm_static, beta_prod_t, alpha_prod_t):
    model_output = self._get_model_output(timestep, num_train_timesteps, norm_inputs_pred, norm_inputs_noise, norm_forcings, norm_static)
    pred_original_sample = (norm_inputs_noise - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


  
  def _ddpm_step(self,
            it: int,
            norm_inputs_pred: xarray.Dataset,
            norm_inputs_noise: xarray.Dataset,
            norm_forcings: xarray.Dataset,
            norm_static: xarray.Dataset,
            noise_scheduler: diffusers.FlaxDDPMScheduler,
            scheduler_state: diffusers.schedulers.scheduling_ddpm_flax.DDPMSchedulerState,
            rng_var: jax.random.KeyArray,
            norm_measurements_diff_interp: Optional[xarray.Dataset] = None,
            **kwargs):
    
    timestep = scheduler_state.timesteps[it]

    # fix timestep embedding in norm_forcings
    # model_output = self._get_model_output(timestep, noise_scheduler.config.num_train_timesteps, norm_inputs_pred, norm_inputs_noise, norm_forcings, norm_static, **kwargs)
    
    #norm_inputs_noise = scheduler_step(noise_scheduler, scheduler_state, model_output, t, jax.random.fold_in(rng_var, int(t)))
    # From: diffusers.FlaxDDPMScheduler.step
    # 1. compute alphas, betas
    alpha_prod_t = scheduler_state.common.alphas_cumprod[timestep]
    # TODO: do we have to use bf16 here?
    alpha_prod_t_prev = jnp.where(timestep > 0, scheduler_state.common.alphas_cumprod[timestep - 1], jnp.array(1.0, dtype=noise_scheduler.dtype))
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    assert noise_scheduler.config.prediction_type == "epsilon"
    #pred_original_sample = (norm_inputs_noise - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    pred_original_sample = self._compute_x0(timestep, noise_scheduler.config.num_train_timesteps, norm_inputs_pred, norm_inputs_noise, norm_forcings, norm_static, beta_prod_t, alpha_prod_t)

    # 3. Clip "predicted x_0"
    if noise_scheduler.config.clip_sample:
        pred_original_sample = xarray_tree.map_structure(lambda x: x.clip(-1, 1), pred_original_sample)

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * scheduler_state.common.betas[timestep]) / beta_prod_t
    current_sample_coeff = scheduler_state.common.alphas[timestep] ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * norm_inputs_noise

    # 6. Add noise
    rng_var_t = jax.random.fold_in(rng_var, timestep)
    stdev = noise_scheduler._get_variance(scheduler_state, timestep) ** 0.5
    
    #if timestep > 0:
    #  norm_inputs_noise = xarray_tree.map_structure(add_noise, pred_prev_sample)
    #else:
    #  norm_inputs_noise = pred_prev_sample
    norm_inputs_noise = jax.lax.cond(timestep > 0, lambda x: xarray_tree.map_structure(functools.partial(add_noise, std_dev=stdev, rng=rng_var_t), x), lambda x: x, pred_prev_sample)

    return norm_inputs_noise

  
  def _repaint_set_timesteps(
      self,
      noise_scheduler: diffusers.FlaxDDPMScheduler,
      scheduler_state: diffusers.schedulers.scheduling_ddpm_flax.DDPMSchedulerState,
      num_inference_steps: int,
      jump_length: int = 10,
      jump_n_sample: int = 10,
  ):
    """
    Sets the discrete timesteps used for the diffusion chain (to be run before inference).

    Args:
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        jump_length (`int`, defaults to 10):
            The number of steps taken forward in time before going backward in time for a single jump (“j” in
            RePaint paper). Take a look at Figure 9 and 10 in the paper.
        jump_n_sample (`int`, defaults to 10):
            The number of times to make a forward time jump for a given chosen time sample. Take a look at Figure 9
            and 10 in the paper.

    """
    num_inference_steps = min(noise_scheduler.config.num_train_timesteps, num_inference_steps)

    timesteps = []

    jumps = {}
    for j in range(0, num_inference_steps - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    t = num_inference_steps
    while t >= 1:
        t = t - 1
        timesteps.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                timesteps.append(t)

    timesteps = jnp.array(timesteps) * (noise_scheduler.config.num_train_timesteps // num_inference_steps)

    return scheduler_state.replace(
      num_inference_steps=num_inference_steps,
      timesteps=timesteps,
    )
  
  def _repaint_step(
        self,
        timestep: int,
        repaint_eta: float,
        repaint_mask: xarray.Dataset,
        norm_measurements_diff_interp: xarray.Dataset,
        norm_inputs_pred: xarray.Dataset,
        norm_inputs_noise: xarray.Dataset,
        norm_forcings: xarray.Dataset,
        norm_static: xarray.Dataset,
        noise_scheduler: diffusers.FlaxDDPMScheduler,
        scheduler_state: diffusers.schedulers.scheduling_ddpm_flax.DDPMSchedulerState,
        rng_var: jax.random.KeyArray,
        norm_measurements_diff_interp_step1: Optional[xarray.Dataset] = None,
        **kwargs
    ) -> xarray.Dataset:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).
    """

    model_output = self._get_model_output(timestep, noise_scheduler.config.num_train_timesteps, norm_inputs_pred, norm_inputs_noise, norm_forcings, norm_static, **kwargs)

    t = timestep
    prev_timestep = timestep - noise_scheduler.config.num_train_timesteps // scheduler_state.num_inference_steps

    # 1. compute alphas, betas
    final_alpha_cumprod = jnp.array(1.0, dtype=scheduler_state.common.alphas_cumprod.dtype)
    alpha_prod_t = scheduler_state.common.alphas_cumprod[t]
    alpha_prod_t_prev = jnp.where(prev_timestep >= 0, scheduler_state.common.alphas_cumprod[prev_timestep], final_alpha_cumprod)
    beta_prod_t = 1 - alpha_prod_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample = (norm_inputs_noise - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

    # 3. Clip "predicted x_0"
    if noise_scheduler.config.clip_sample:
        pred_original_sample = xarray_tree.map_structure(lambda x: x.clip(-1, 1), pred_original_sample)

    # We choose to follow RePaint Algorithm 1 to get x_{t-1}, however we
    # substitute formula (7) in the algorithm coming from DDPM paper
    # (formula (4) Algorithm 2 - Sampling) with formula (12) from DDIM paper.
    # DDIM schedule gives the same results as DDPM with eta = 1.0
    # Noise is being reused in 7. and 8., but no impact on quality has
    # been observed.

    # 5. Add noise
    # Compute variance
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    # For t > 0, compute predicted variance βt (see formula (6) and (7) from
    # https://arxiv.org/pdf/2006.11239.pdf) and sample from it to get
    # previous sample x_{t-1} ~ N(pred_prev_sample, variance) == add
    # variance to pred_sample
    # Is equivalent to formula (16) in https://arxiv.org/pdf/2010.02502.pdf
    # without eta.
    # variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    std_dev_t = repaint_eta * variance ** 0.5
    rng_var_t = jax.random.fold_in(rng_var, timestep)

    # 6. compute "direction pointing to x_t" of formula (12)
    # from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * model_output

    # 7. compute x_{t-1} of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_unknown_part = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
    if repaint_eta > 0:
      prev_unknown_part = jax.lax.cond(t > 0, lambda x: xarray_tree.map_structure(functools.partial(add_noise, std_dev=std_dev_t, rng=rng_var_t), x), 
                                      lambda x: x, prev_unknown_part)

    # 8. Algorithm 1 Line 5 https://arxiv.org/pdf/2201.09865.pdf
    prev_known_part = xarray_tree.map_structure(functools.partial(add_noise, std_dev=((1 - alpha_prod_t_prev) ** 0.5), rng=rng_var_t), (alpha_prod_t_prev**0.5) * norm_measurements_diff_interp)

    # 9. Algorithm 1 Line 8 https://arxiv.org/pdf/2201.09865.pdf
    pred_prev_sample = repaint_mask * prev_known_part + (1.0 - repaint_mask) * prev_unknown_part

    return pred_prev_sample

  def _repaint_undo_step(self, timestep: int, 
                sample: xarray.Dataset, 
                noise_scheduler: diffusers.FlaxDDPMScheduler,
                scheduler_state: diffusers.schedulers.scheduling_ddpm_flax.DDPMSchedulerState,
                rng_var: jax.random.KeyArray):
    n = noise_scheduler.config.num_train_timesteps // scheduler_state.num_inference_steps
    rng_undo = jax.random.split(rng_var, 1)[0]
    rng_undo_t = jax.random.fold_in(rng_undo, timestep)
    for i in range(n):
      rng_undo_ti = jax.random.fold_in(rng_undo_t, i)
      beta = scheduler_state.common.betas[timestep + i]

      # 10. Algorithm 1 Line 10 https://arxiv.org/pdf/2201.09865.pdf
      sample = xarray_tree.map_structure(functools.partial(add_noise, std_dev=beta**0.5, rng=rng_undo_ti), (1 - beta) ** 0.5 * sample)
    return sample

  def repaint_forward(self,
                repaint_mask: xarray.DataArray,
                norm_measurements_diff_interp: xarray.Dataset,
                norm_inputs_pred: xarray.Dataset,
                norm_forcings: xarray.Dataset,
                norm_static: xarray.Dataset,
                noise_scheduler: diffusers.FlaxDDPMScheduler,
                scheduler_state: diffusers.schedulers.scheduling_ddpm_flax.DDPMSchedulerState,
                num_inference_steps: int,
                rng_batch: jax.random.KeyArray,
                repaint_eta: float = 0.0,
                repaint_jump_length: int = 10,
                repaint_jump_n_sample: int = 10,
                progress_bar: bool = False,
                **kwargs
                ) -> xarray.Dataset:
    scheduler_state = self._repaint_set_timesteps(noise_scheduler, scheduler_state, num_inference_steps, repaint_jump_length, repaint_jump_n_sample)
    rng_init, rng_var = jax.random.split(rng_batch)
    norm_inputs_noise = xarray_tree.map_structure(lambda x: xarray.zeros_like(x) + jax.random.normal(jax.random.fold_in(rng_init, hash(x.name)), x.shape, x.dtype), norm_inputs_pred)

    repaint_step_fn = lambda timestep, norm_inputs_noise: self._repaint_step(timestep, repaint_eta, repaint_mask, 
                                                          norm_measurements_diff_interp, norm_inputs_pred, 
                                                          norm_inputs_noise, norm_forcings, norm_static, 
                                                          noise_scheduler, scheduler_state, rng_var, **kwargs)
    repaint_undo_step_fn = lambda timestep, norm_inputs_noise: self._repaint_undo_step(timestep, norm_inputs_noise, 
                                                                    noise_scheduler, scheduler_state, rng_var)
    repaint_step_jitted = jax.jit(repaint_step_fn, donate_argnames=["norm_inputs_noise"])
    repaint_undo_step_jitted = jax.jit(repaint_undo_step_fn, donate_argnames=["norm_inputs_noise"])

    t_last = scheduler_state.timesteps[0] + 1
    pbar = scheduler_state.timesteps
    if progress_bar:
      pbar = tqdm(pbar, desc="RePaint")
    for timestep in pbar:
      if timestep < t_last:
        norm_inputs_noise = repaint_step_jitted(timestep, norm_inputs_noise)
      else:
        norm_inputs_noise = repaint_undo_step_jitted(timestep, norm_inputs_noise)
      assert jnp.isnan(xarray_jax.unwrap_data(norm_inputs_noise["geopotential"])).all() == False
      t_last = timestep
    norm_res_preditions = norm_inputs_noise
    inputs_pred = unnormalize(norm_inputs_pred, self._scales, self._locations)
    return xarray_tree.map_structure(
        lambda res: self._unnormalize_prediction_and_add_input(inputs_pred, res),
        norm_res_preditions)



  def __call__(self,
               norm_inputs_pred: xarray.Dataset,
               norm_forcings: xarray.Dataset,
               norm_static: xarray.Dataset,
               noise_scheduler: diffusers.FlaxDDPMScheduler,
               scheduler_state: diffusers.schedulers.scheduling_ddpm_flax.DDPMSchedulerState,
               num_inference_steps: int,
               rng_batch: jax.random.KeyArray,
               progress_bar: bool = False,
               **kwargs
               ) -> xarray.Dataset:
    
    scheduler_state = noise_scheduler.set_timesteps(scheduler_state, num_inference_steps)
    rng_init, rng_var = jax.random.split(rng_batch)
    norm_inputs_noise = xarray_tree.map_structure(lambda x: xarray.zeros_like(x) + jax.random.normal(jax.random.fold_in(rng_init, hash(x.name)), x.shape, x.dtype), norm_inputs_pred)
    
    loop_fn = lambda it, norm_inputs_noise: self._ddpm_step(it = it,
                                          norm_inputs_noise=norm_inputs_noise,
                                          norm_inputs_pred=norm_inputs_pred, 
                                          norm_forcings=norm_forcings, 
                                          norm_static=norm_static, 
                                          noise_scheduler=noise_scheduler, 
                                          scheduler_state=scheduler_state, 
                                          rng_var=rng_var, **kwargs)
    # TODO: move this to outer level
    loop_fn_jitted = jax.jit(loop_fn, donate_argnames=["norm_inputs_noise"])
    
    pbar = range(len(scheduler_state.timesteps))
    if progress_bar:
      pbar = tqdm(pbar, desc="DDPM Inference")
    for it in pbar:
      norm_inputs_noise = loop_fn_jitted(it, norm_inputs_noise)
    norm_res_preditions = norm_inputs_noise
    #norm_res_preditions = jax.lax.fori_loop(0, len(scheduler_state.timesteps), loop_fn_jitted, norm_inputs_noise)

    inputs_pred = unnormalize(norm_inputs_pred, self._scales, self._locations)
    return xarray_tree.map_structure(
        lambda res: self._unnormalize_prediction_and_add_input(inputs_pred, res),
        norm_res_preditions)

  def loss(self,
           norm_inputs_noise: xarray.Dataset,
           norm_inputs_pred: xarray.Dataset,
           targets_noise: xarray.Dataset,
           norm_forcings: xarray.Dataset,
           norm_static: xarray.Dataset,
           **kwargs,
           ) -> predictor_base.LossAndDiagnostics:
    """Returns the loss computed on normalized inputs and targets."""
    norm_inputs = xarray.concat([norm_inputs_noise, norm_inputs_pred], dim='time')
    norm_inputs = xarray.merge([norm_inputs, xarray.concat([norm_forcings, norm_forcings], dim='time'), norm_static]) # inputs includes forcings as well
    return self._predictor.loss(
        norm_inputs, targets_noise, forcings=norm_forcings, **kwargs)