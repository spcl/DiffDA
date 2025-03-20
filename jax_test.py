# save as test_jax.py
import jax
print(f"JAX version: {jax.__version__}")
try:
    print(f"Available devices: {jax.devices()}")
except Exception as e:
    print(f"Error getting devices: {e}")