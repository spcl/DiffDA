# Setup environment
Follow `Dockerfile` to setup a container environment for NVIDIA GPUs.

# Prepare forecast data from GraphCast
Setup a Google Cloud Storage key named `gcs_key.json` and put it at `GraphCast_src/`

Execute `python3 GraphCast_src/graphcast_runner.py --resolution .25 --pressure_levels 13 --autoregressive_steps 8 --test_year_start 1977 --test_year_end 2016` and `python3 GraphCast_src/graphcast_runner.py --resolution .25 --pressure_levels 13 --autoregressive_steps 1 --test_year_start 1977 --test_year_end 2016` to generate 48h and 6h GraphCast forecast data need for training the diffusion model.

# Prepare GraphCast weights and parameters
Visit: https://console.cloud.google.com/storage/browser/dm_graphcast and download `GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz` under `params` folder and the whole `stats` folder.

# Train diffusion model
Execute `python3 DiffDA/train_conditional_graphcast.py` and pass in required arguments (hyperparameters, ERA5, forecast data path, etc.)

# Run data assimilation
- Execute `python3 DiffDA/inference_data_assimilation.py --num_autoregressive_steps=1 ...` to run single step data assimilation
- Execute `python3 DiffDA/inference_data_assimilation.py --num_autoregressive_steps=n ...` (n > 2) to run autoregressive data assimilation
- Execute `python3 DiffDA/inference_data_assimilation_gc.py --num_autoregressive_steps=n ...` to run (autoregressive) GraphCast forecast on single step assimilated data

# Implementation detail
- `GraphCast_src/graphcast/normalization.py`: ddpm and repaint algorithm for inference
- `DiffDA/train_conditional_graphcast.py`: training diffusion model with GraphCast as backbone
- `DiffDA/inference_data_assimilation.py`: run single step & autoregressive data assimilation
- `DiffDA/inference_data_assimilation_gc.py`: run GraphCast forecast on single step assimilated data