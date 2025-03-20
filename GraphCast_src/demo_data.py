import xarray as xr
from pathlib import Path
import os

def load_normalization(bucket=None, cache_path=None):
    """
    直接从硬编码的本地路径加载归一化统计数据
    
    Args:
        bucket: 忽略此参数，保留只是为了兼容性
        cache_path: 忽略此参数，保留只是为了兼容性
    
    Returns:
        三个归一化相关的数据集
    """
    # 使用硬编码的路径
    local_path = Path("/sharefiles4/qubohuan/Projects/DiffDA_Low/Data")
    
    print(f"Loading normalization data from fixed path: {local_path}")
    
    # 定义文件名
    files = {
        "diffs": "stats_diffs_stddev_by_level.nc",
        "mean": "stats_mean_by_level.nc", 
        "stddev": "stats_stddev_by_level.nc"
    }
    
    # 构建完整文件路径
    diffs_path = local_path / files["diffs"]
    mean_path = local_path / files["mean"]
    stddev_path = local_path / files["stddev"]
    
    # 检查文件是否存在
    if not all(p.exists() for p in [diffs_path, mean_path, stddev_path]):
        print(f"Warning: Not all normalization files exist at {local_path}")
        print(f"diffs path exists: {diffs_path.exists()}")
        print(f"mean path exists: {mean_path.exists()}")
        print(f"stddev path exists: {stddev_path.exists()}")
        
        # 列出目录中的文件以帮助调试
        print("Files in directory:")
        for file in local_path.iterdir():
            print(f" - {file.name}")
        
        raise FileNotFoundError(f"Normalization files not found in {local_path}")
    
    # 加载文件
    print(f"Loading files: {diffs_path}, {mean_path}, {stddev_path}")
    try:
        diffs_stddev_by_level = xr.load_dataset(str(diffs_path)).compute()
        mean_by_level = xr.load_dataset(str(mean_path)).compute()
        stddev_by_level = xr.load_dataset(str(stddev_path)).compute()
        print("Successfully loaded normalization data")
    except Exception as e:
        print(f"Error loading normalization files: {e}")
        raise
    
    return diffs_stddev_by_level, mean_by_level, stddev_by_level