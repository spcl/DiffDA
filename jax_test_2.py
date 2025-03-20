import os
# 尝试不同的 XLA 标志
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_cuda_data_dir=/path/to/cuda"

import jax
from jax.lib import xla_bridge
print(f"JAX version: {jax.__version__}")

# 以不同方式初始化 JAX 平台
try:
    platform = xla_bridge.get_backend().platform
    print(f"JAX platform: {platform}")
    
    # 尝试这种方式获取设备
    from jax.lib import xla_client
    local_devices = xla_client.get_local_device_count()
    print(f"Local device count: {local_devices}")
    
    # 最后尝试获取设备列表
    print(f"Devices: {jax.devices()}")
except Exception as e:
    print(f"Error: {e}")