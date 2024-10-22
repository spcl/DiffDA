import dataclasses
import pickle
from typing import Union

import haiku as hk
import optax
from graphcast import graphcast, checkpoint
import jax
from pathlib import Path
import os
import re
import diffusers

@dataclasses.dataclass(frozen=True)
class TrainingCheckpoint:
    params: hk.Params
    opt_state: optax.OptState
    scheduler_state: diffusers.schedulers.scheduling_ddpm_flax.DDPMSchedulerState
    task_config: graphcast.TaskConfig
    model_config: graphcast.ModelConfig
    epoch: int
    rng: jax.Array
    num_train_timesteps: int = 1000
# TODO: add num_train_timesteps

def save_checkpoint(directory: Path, ckpt: TrainingCheckpoint) -> None:
    """
    Stores a checkpoint at the given directory with the name
    directory / diff_gc_{epoch}.npz
    If the given directory does not exist, it will be created.
    If there already exists a checkpoint in the same directory with the same epoch,
    it will be overwritten.
    """
    directory.mkdir(parents=True, exist_ok=True)
    save_file = directory / f"diff_gc_{ckpt.epoch}.npz"
    with open(save_file, mode="wb") as file:
        pickle.dump(ckpt, file)
        #checkpoint.dump(file, ckpt)

def load_checkpoint(directory: Path, epoch: int = -1) -> Union[TrainingCheckpoint, None]:
    """
    Loads a checkpoint from the given directory. If a non-negative epoch is given,
    that checkpoint is loaded. Otherwise, the latest epoch is loaded.
    If no checkpoint is found, None is returned.
    """
    if not directory.exists():
        return None
    # Define the pattern using a regular expression
    pattern = r"diff_gc_(\d+)"
    # Initialize an empty list to store the tuples (i, path)
    file_tuples = []
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        match = re.match(pattern, filename)
        if match:
            # Extract the epoch i from the match
            i = int(match.group(1))
            # Construct the full path to the file
            filepath = os.path.join(directory, filename)
            # Append the tuple (i, path) to the list
            file_tuples.append((i, filepath))

    if len(file_tuples) == 0:
        return None

    # Sort the list of tuples based on the integer i
    file_tuples.sort()
    ckpt_path = None
    if epoch < 0:
        ckpt_path = file_tuples[-1][1]
    else:
        for (i, f) in file_tuples:
            if i == epoch:
                ckpt_path = f
                break

    with open(ckpt_path, "rb") as file:
        return pickle.load(file)
        #return checkpoint.load(file, TrainingCheckpoint)