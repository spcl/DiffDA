#!/usr/bin/env python
# coding: utf-8

# # GraphCast
# 
# This colab lets you run several versions of GraphCast.
# 
# The model weights, normalization statistics, and example inputs are available on [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast).
# 
# A Colab runtime with TPU/GPU acceleration will substantially speed up generating predictions and computing the loss/gradients. If you're using a CPU-only runtime, you can switch using the menu "Runtime > Change runtime type".

# > <p><small><small>Copyright 2023 DeepMind Technologies Limited.</small></p>
# > <p><small><small>Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at <a href="http://www.apache.org/licenses/LICENSE-2.0">http://www.apache.org/licenses/LICENSE-2.0</a>.</small></small></p>
# > <p><small><small>Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.</small></small></p>

# Use most memory-conservative allocation scheme
# See https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
# Either set this in your environment or set this before any import of jax code
import os
from pathlib import Path

import graphcast_wrapper
from weatherbench2_dataloader import WeatherBench2Dataset

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import dataclasses
import functools
import time

from graphcast import losses
from graphcast import data_utils
from graphcast import graphcast
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import numpy as np

from buckets import authenticate_bucket


def train_model(loss_fn: hk.TransformedWithState, params, train_inputs, train_targets, train_forcings, epochs=5):
    """
    @param loss_fn: a hk_transform wrapped loss function (encapsulates the model as well)
    @param params: initial model parameters (from a checkpoint)
    @param train_inputs
    @param train_targets
    @param train_forcings
    @param epochs
    """
    assert epochs is not None

    def grads_fn(params, state, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(params,
                                                            state,
                                                            jax.random.PRNGKey(0),
                                                            i,
                                                            t,
                                                            f)
            return loss, (diagnostics, next_state)

        # TODO add reduce_axes=('batch',)
        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(params,
                                                                                          state,
                                                                                          inputs,
                                                                                          targets,
                                                                                          forcings)
        return loss, diagnostics, next_state, grads

    grads_fn_jitted = jax.jit(grads_fn)

    runtimes = []
    for i in range(epochs):
        tic = time.perf_counter()
        # Gradient computation (backprop through time)
        loss, diagnostics, next_state, grads = grads_fn_jitted(
            params=params,
            state={},
            inputs=train_inputs,
            targets=train_targets,
            forcings=train_forcings)
        jax.block_until_ready(grads)
        jax.block_until_ready(loss)
        mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
        print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")
        toc = time.perf_counter()
        print(f"Step {i} took {toc-tic}s")
        if i > 0:
            runtimes.append(toc-tic)
    print("Training step time: ", np.mean(np.asarray(runtimes)), " +-", np.std(np.asarray(runtimes)))


def evaluate_model(fwd_cost_fn, task_config: graphcast.TaskConfig, dataloader, autoregressive_steps=1):
    """
    Perform inference using the given forward cost function.
    Assumes each autoregressive step has a lead time of 6 hours.
    @param fwd_cost_fn Cost function that takes inputs, targets, and forcings to return a scalar cost.
    @param task_config Corresponding task configuration (must match model underlying the fwd_cost_fn)
    @param dataloader must support __len__ and __getitem__. Each batch should be an xarray.Dataarray
    @param autoregressive_steps how many times to unroll the model. Must match with the batch provided from the dataloader.
    """
    costs = []
    for i in range(len(dataloader)):
        batch = dataloader[i]

        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(batch,
                                                                               target_lead_times=f"{autoregressive_steps * 6}h",
                                                                               # Note that this sets input duration to 12h
                                                                               **dataclasses.asdict(task_config))

        cost = fwd_cost_fn(inputs, targets, forcings)
        print(f"Batch {i}/{len(dataloader)} cost: {cost}")
        costs.append(cost)

    mean_cost = np.asarray(costs).mean()
    print("Mean cost:", mean_cost, " +-", np.asarray(costs).std())
    return mean_cost

def rmse_forward(inputs, targets, forcings, forward_fn, level: int, variable_weights: dict[str, float]) -> float:
    """
    Compute the RMSE given a forward function
    @param inputs
    @param targets
    @param forcings
    @param foward_fn Function that computes the prediction. Will be rolled out autoregressively.
    @param level Pressure level to evaluate the RMSE on
    @param variable_weights weight to assign to each target variable
    """
    predictions = rollout.chunked_prediction(
        forward_fn,
        rng=jax.random.PRNGKey(353),
        inputs=inputs,
        targets_template=targets * np.nan,
        forcings=forcings)
    def mse(x, y):
        return (x-y) ** 2

    mse_error, _ = losses.weighted_error_per_level(predictions,
                                                   targets,
                                                   variable_weights,
                                                   mse,
                                                   functools.partial(losses.single_level_weights, level=level))
    return np.sqrt(mse_error.mean().item())

def main(resolution: float = 0.25,
         pressure_levels: int = 13,
         autoregressive_steps: int = 1,
         test_years=None,
         test_variable: str = 'geopotential',
         test_pressure_level: int = 500,
         repetitions: int = 5) -> None:
    """
    resolution: resolution of the model in degrees
    pressure_levels: number of pressure levels
    train: If true, computes gradients of the model (currently does not actually train anything)
    autoregressive_steps: How many rollout steps to perform. If 1, a single-step prediction is done.
    repetitions: For time measurement purposes, how many repetition are done.
    """
    #
    # - **Source**: era5, hres
    # - **Resolution**: 0.25deg, 1deg
    # - **Levels**: 13, 37
    #
    # Not all combinations are available.
    # - HRES is only available in 0.25 deg, with 13 pressure levels.

    if test_years is None:
        test_years = [2016]
    data_path = os.environ.get('DATA_PATH')

    run_forward, checkpoint = graphcast_wrapper.retrieve_model(resolution, pressure_levels, Path(data_path))

    # Always pass params so the usage below are simpler
    def with_params(fn):
        return functools.partial(fn, params=checkpoint.params)

    # # Run the model (Inference)
    dataset = WeatherBench2Dataset(
                            year=test_years[0],
                            steps=autoregressive_steps,
                            steps_per_input=3)

    # Compile the forward function and add the configuration and params as partials
    run_forward_jitted = with_params(jax.jit(run_forward.apply))

    # Pick the test_variable as the only one
    variable_weights = {var: 1 if var == test_variable else 0 for var in checkpoint.task_config.target_variables}
    # We create the loss function by passing the forward function and parameters as a partial to rmse_forward
    loss_function = functools.partial(rmse_forward,
                                        forward_fn = run_forward_jitted,
                                        level=test_pressure_level,
                                        variable_weights=variable_weights)
    evaluate_model(loss_function, checkpoint.task_config, dataset, autoregressive_steps=autoregressive_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference and training showcase for Graphcast.')

    # Add the arguments with default values
    parser.add_argument('--resolution', type=float, default=0.25, help='Resolution of the graph in the model.')
    parser.add_argument('--pressure_levels', type=int, default=13, help='Number of pressure levels in the model.')
    parser.add_argument('--autoregressive_steps', type=int, default=1, help='Number of time steps to predict into the future.')
    parser.add_argument('--test_year_start', type=int, default=2016, help='First year to use for testing (inference).')
    parser.add_argument('--test_year_end', type=int, default=2016, help='Last year to use for testing (inference).')
    parser.add_argument('--test_pressure_level', type=int, default=500, help='Pressure level to use for testing (inference).')
    parser.add_argument('--test_variable', type=str, default='geopotential', help='Variable to use for testing (inference).')
    parser.add_argument('--prediction_store_path', type=str, default=None, help='If not none, evaluate predictions and store them here.')


    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments & call main
    main(args.resolution,
         args.pressure_levels,
         args.autoregressive_steps,
         list(range(args.test_year_start, args.test_year_end+1)),
         args.test_variable,
         args.test_pressure_level)
