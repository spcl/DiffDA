export DATA_PATH=/sharefiles4/qubohuan/Projects/DiffDA_Low/Data
# export XLA_PYTHON_CLIENT_MEM_FRACTION="0.8"  # 尝试增加到0.8
# 重置之前的设置
unset XLA_PYTHON_CLIENT_MEM_FRACTION
unset XLA_PYTHON_CLIENT_PREALLOCATE
unset XLA_PYTHON_CLIENT_ALLOCATOR

# 使用更稳定的设置
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export JAX_TRACEBACK_FILTERING=off  # 显示完整的JAX错误追踪
export JAX_DISABLE_JIT=False  # 确保JIT启用
export JAX_PLATFORM_NAME="gpu"  # 明确使用GPU

python3 GraphCast_src/graphcast_runner.py --resolution 1 --pressure_levels 13 --autoregressive_steps 8 --test_year_start 2006 --test_year_end 2016
# python3 GraphCast_src/graphcast_runner.py --resolution 1 --pressure_levels 13 --autoregressive_steps 1 --test_year_start 1979 --test_year_end 2016