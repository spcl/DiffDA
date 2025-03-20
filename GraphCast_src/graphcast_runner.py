#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import graphcast_wrapper
from weatherbench2_dataloader import WeatherBench2Dataset
import sys
import traceback
import psutil  # 用于监控内存使用
import gc    # 垃圾回收
import logging  # 用于日志记录

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("graphcast_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 内存使用监控函数
def log_memory_usage(tag):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"MEMORY [{tag}] - RSS: {memory_info.rss / (1024**3):.2f} GB, VMS: {memory_info.vms / (1024**3):.2f} GB")
    
# 异常处理装饰器
def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"开始执行函数: {func.__name__}")
            log_memory_usage(f"开始 {func.__name__}")
            result = func(*args, **kwargs)
            log_memory_usage(f"结束 {func.__name__}")
            logger.info(f"成功执行函数: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行出错: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"  # 限制内存使用比例

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
from tqdm import tqdm

from buckets import authenticate_bucket
from demo_data import load_normalization

@exception_handler
def train_model(loss_fn: hk.TransformedWithState, params, train_inputs, train_targets, train_forcings, epochs=5):
    """
    训练模型函数（已添加异常处理和内存监控）
    """
    assert epochs is not None

    def grads_fn(params, state, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            try:
                logger.info("开始计算损失函数")
                log_memory_usage("损失函数计算前")
                (loss, diagnostics), next_state = loss_fn.apply(params,
                                                              state,
                                                              jax.random.PRNGKey(0),
                                                              i,
                                                              t,
                                                              f)
                log_memory_usage("损失函数计算后")
                logger.info("损失函数计算完成")
                return loss, (diagnostics, next_state)
            except Exception as e:
                logger.error(f"损失函数计算失败: {e}")
                logger.error(traceback.format_exc())
                raise

        logger.info("开始计算梯度")
        log_memory_usage("梯度计算前")
        try:
            (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(params,
                                                                                            state,
                                                                                            inputs,
                                                                                            targets,
                                                                                            forcings)
            log_memory_usage("梯度计算后")
            logger.info("梯度计算完成")
            return loss, diagnostics, next_state, grads
        except Exception as e:
            logger.error(f"梯度计算失败: {e}")
            logger.error(traceback.format_exc())
            raise

    try:
        logger.info("开始JIT编译梯度函数")
        grads_fn_jitted = jax.jit(grads_fn)
        logger.info("JIT编译完成")
    except Exception as e:
        logger.error(f"JIT编译失败: {e}")
        logger.error(traceback.format_exc())
        raise

    runtimes = []
    progress_bar = tqdm(range(epochs), desc="训练进度")
    for i in progress_bar:
        try:
            tic = time.perf_counter()
            logger.info(f"开始第 {i+1}/{epochs} 次训练迭代")
            log_memory_usage(f"迭代 {i+1} 前")
            
            # 梯度计算
            loss, diagnostics, next_state, grads = grads_fn_jitted(
                params=params,
                state={},
                inputs=train_inputs,
                targets=train_targets,
                forcings=train_forcings)
            
            jax.block_until_ready(grads)
            jax.block_until_ready(loss)
            
            mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
            toc = time.perf_counter()
            
            log_memory_usage(f"迭代 {i+1} 后")
            logger.info(f"完成第 {i+1}/{epochs} 次训练迭代, 损失: {loss:.4f}, 平均梯度: {mean_grad:.6f}, 用时: {toc-tic:.2f}s")
            
            # 更新进度条信息
            progress_bar.set_postfix({"Loss": f"{loss:.4f}", "Mean |grad|": f"{mean_grad:.6f}", "Time": f"{toc-tic:.2f}s"})
            
            if i > 0:
                runtimes.append(toc-tic)
                
            # 主动触发垃圾回收
            gc.collect()
            
        except Exception as e:
            logger.error(f"迭代 {i+1} 失败: {e}")
            logger.error(traceback.format_exc())
            raise
    
    if runtimes:
        logger.info(f"训练步骤平均用时: {np.mean(np.asarray(runtimes)):.4f} ± {np.std(np.asarray(runtimes)):.4f}")

@exception_handler
def evaluate_model(fwd_cost_fn, task_config: graphcast.TaskConfig, dataloader, autoregressive_steps=1):
    """
    Perform inference using the given forward cost function.
    """
    costs = []
    progress_bar = tqdm(range(len(dataloader)), desc="评估进度")
    
    for i in progress_bar:
        try:
            logger.info(f"评估样本 {i+1}/{len(dataloader)}")
            log_memory_usage(f"获取批次 {i+1} 前")
            
            batch = dataloader[i]
            log_memory_usage(f"获取批次 {i+1} 后")
            
            logger.info(f"开始提取输入、目标和强制项")
            inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                batch,
                target_lead_times=f"{autoregressive_steps * 6}h",
                **dataclasses.asdict(task_config)
            )
            log_memory_usage(f"提取输入、目标和强制项后")
            
            logger.info(f"开始计算损失")
            cost = fwd_cost_fn(inputs, targets, forcings)
            log_memory_usage(f"计算损失后")
            
            costs.append(cost)
            logger.info(f"样本 {i+1} 损失: {cost:.4f}")
            
            # 更新进度条信息
            progress_bar.set_postfix({"Cost": f"{cost:.4f}"})
            
            # 主动触发垃圾回收
            gc.collect()
            
        except Exception as e:
            logger.error(f"评估样本 {i+1} 失败: {e}")
            logger.error(traceback.format_exc())
            raise

    if costs:
        mean_cost = np.asarray(costs).mean()
        std_cost = np.asarray(costs).std()
        logger.info(f"平均损失: {mean_cost:.4f} ± {std_cost:.4f}")
    else:
        logger.warning("未计算任何损失")
        
    return mean_cost if costs else None

@exception_handler
def rmse_forward(inputs, targets, forcings, forward_fn, level: int, variable_weights: dict[str, float]) -> float:
    """
    计算RMSE（已添加异常处理和内存监控）
    """
    try:
        logger.info("开始执行预测")
        log_memory_usage("预测前")
        
        predictions = rollout.chunked_prediction(
            forward_fn,
            rng=jax.random.PRNGKey(353),
            inputs=inputs,
            targets_template=targets * np.nan,
            forcings=forcings)
            
        log_memory_usage("预测后")
        logger.info("预测完成")
        
        def mse(x, y):
            return (x-y) ** 2
        
        logger.info("开始计算加权误差")
        mse_error, _ = losses.weighted_error_per_level(
            predictions,
            targets,
            variable_weights,
            mse,
            functools.partial(losses.single_level_weights, level=level))
            
        log_memory_usage("误差计算后")
        logger.info("误差计算完成")
        
        rmse_value = np.sqrt(mse_error.mean().item())
        logger.info(f"RMSE: {rmse_value:.4f}")
        return rmse_value
        
    except Exception as e:
        logger.error(f"RMSE计算失败: {e}")
        logger.error(traceback.format_exc())
        raise

@exception_handler
def main(resolution: float = 0.25,
         pressure_levels: int = 13,
         autoregressive_steps: int = 1,
         test_years=None,
         test_variable: str = 'geopotential',
         test_pressure_level: int = 500,
         repetitions: int = 5) -> None:
    """
    主函数（已添加异常处理和内存监控）
    """
    if test_years is None:
        test_years = [2016]
        
    logger.info(f"开始执行GraphCast, 参数: 分辨率={resolution}, 压力层={pressure_levels}, 自回归步数={autoregressive_steps}")
    logger.info(f"测试年份: {test_years}, 测试变量: {test_variable}, 测试压力层: {test_pressure_level}")
    
    data_path = os.environ.get('DATA_PATH')
    logger.info(f"数据路径: {data_path}")
    
    try:
        log_memory_usage("加载模型前")
        logger.info("开始加载模型")
        
        run_forward, checkpoint = graphcast_wrapper.retrieve_model(resolution, pressure_levels, Path(data_path))
        
        log_memory_usage("加载模型后")
        logger.info("模型加载完成")
        
        # Always pass params so the usage below are simpler
        def with_params(fn):
            return functools.partial(fn, params=checkpoint.params)
        
        # 加载数据集
        logger.info(f"开始加载数据集: 年份={test_years[0]}, 步数={autoregressive_steps}")
        log_memory_usage("加载数据集前")
        
        dataset = WeatherBench2Dataset(
                                year=test_years[0],
                                steps=autoregressive_steps,
                                steps_per_input=3)
                                
        log_memory_usage("加载数据集后")
        logger.info(f"数据集加载完成, 包含 {len(dataset)} 个样本")

        # JIT编译前向函数
        logger.info("开始JIT编译前向函数")
        log_memory_usage("JIT编译前")
        
        run_forward_jitted = with_params(jax.jit(run_forward.apply))
        
        log_memory_usage("JIT编译后")
        logger.info("JIT编译完成")

        # 设置变量权重
        variable_weights = {var: 1 if var == test_variable else 0 for var in checkpoint.task_config.target_variables}
        logger.info(f"设置变量权重: {variable_weights}")
        
        # 创建损失函数
        logger.info("创建损失函数")
        loss_function = functools.partial(rmse_forward,
                                            forward_fn=run_forward_jitted,
                                            level=test_pressure_level,
                                            variable_weights=variable_weights)
                                            
        # 评估模型
        logger.info("开始评估模型")
        log_memory_usage("评估模型前")
        
        evaluate_model(loss_function, checkpoint.task_config, dataset, autoregressive_steps=autoregressive_steps)
        
        log_memory_usage("评估模型后")
        logger.info("模型评估完成")
        
    except Exception as e:
        logger.error(f"主函数执行失败: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        logger.info("程序开始执行")
        log_memory_usage("程序开始")
        
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
        logger.info(f"命令行参数: {args}")

        # 记录一下环境信息
        logger.info(f"Python版本: {sys.version}")
        logger.info(f"JAX版本: {jax.__version__}")
        try:
            logger.info(f"可用设备: {jax.devices()}")
        except:
            logger.warning("无法获取JAX设备信息")
        
        # Access the arguments & call main
        main(args.resolution,
            args.pressure_levels,
            args.autoregressive_steps,
            list(range(args.test_year_start, args.test_year_end+1)),
            args.test_variable,
            args.test_pressure_level)
            
        logger.info("程序成功执行完毕")
        log_memory_usage("程序结束")
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)