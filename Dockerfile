FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv git curl wget libeccodes0 tensorrt libnvinfer-dev libnvinfer-plugin-dev

# 安装 Python 依赖
RUN pip install --upgrade pip && pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install cartopy chex colabtools dask dm-haiku jraph matplotlib numpy \
    pandas rtree scipy trimesh typing_extensions xarray \
    ipython ipykernel jupyterlab notebook ipywidgets google-cloud-storage dm-tree \
    cfgrib zarr gcsfs dask jax-dataloader tensorflow tensorboard-plugin-profile nvtx \
    wandb
RUN pip install --upgrade diffusers[flax]

ENV TF_CPP_MIN_LOG_LEVEL=2

# 设置工作目录
WORKDIR /workspace

CMD [ "/bin/bash" ]