import functools
import inspect
import os
from typing import Any, Callable, Dict, Optional
from tensorguard.core.engine import ShapeEngine

def validate(
    returns: Optional[str] = None, 
    dtypes: Optional[Dict[str, str]] = None,
    **shape_kwargs
) -> Callable:
    """
    TensorFlow specific decorator for validating tensor shapes.
    Supports dynamic/symbolic dimensions (None) in Graph mode.
    
    Args:
        returns: Expected shape of the return value.
        dtypes: Optional dict mapping args to TF dtypes (e.g. {"images": "float32"}).
    """
    def decorator(func: Callable) -> Callable:
        if os.environ.get("TENSORGUARD_ENV") == "production":
            return func
            
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            engine = ShapeEngine()
            
            # Using lazy import to not slow down non-TF users
            import tensorflow as tf
            
            def check_tensor(name: str, tensor: Any, expected: str):
                # Using hasattr/getattr to safely check type without isinstance to allow mocking in tests 
                # but standard practice is isinstance
                if not isinstance(tensor, tf.Tensor):
                    raise TypeError(f"TensorFlow Validator: Argument '{name}' must be a tf.Tensor.")
                
                # Convert tf.TensorShape to a tuple, replace None with -1 for the engine to skip
                try:
                    shape_list = tensor.shape.as_list()
                    actual_shape = tuple(dim if dim is not None else -1 for dim in shape_list)
                except ValueError:
                    # Shape is completely unknown (e.g., tf.Tensor(shape=<unknown>))
                    actual_shape = tuple([-1] * len(expected.split()))
                    
                # Dtype check
                if dtypes is not None and name in dtypes:
                    expected_dtype = dtypes[name]
                    actual_dtype = tensor.dtype.name
                    if actual_dtype != expected_dtype:
                        raise TypeError(
                            f"TensorFlow Validator: DType mismatch for '{name}'. "
                            f"Expected '{expected_dtype}', got '{actual_dtype}'."
                        )
                        
                engine.match_shape(name, actual_shape, expected)

            # 1. Validate inputs
            for arg_name, expected_shape in shape_kwargs.items():
                if arg_name in bound_args.arguments:
                    arg_value = bound_args.arguments[arg_name]
                    check_tensor(arg_name, arg_value, expected_shape)
                    
            # 2. Execute original function
            result = func(*args, **kwargs)
            
            # 3. Validate return value
            if returns is not None:
                check_tensor("return_value", result, returns)
                
            return result
            
        return wrapper
    return decorator
