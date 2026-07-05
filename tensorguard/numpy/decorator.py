import functools
import inspect
import os
from typing import Any, Callable, Dict, Optional, Union, Tuple, List
from tensorguard.core.engine import ShapeEngine

def validate(
    returns: Optional[Union[str, Tuple[str, ...], List[str], Dict[str, str]]] = None, 
    dtypes: Optional[Dict[str, str]] = None,
    **shape_kwargs
) -> Callable:
    """
    NumPy specific decorator for validating tensor shapes.
    
    Usage:
        @validate(images="b c h w", labels="b", returns="b classes")
        def forward(images, labels):
            ...
            
    Args:
        returns: Expected shape of the return value.
        dtypes: Optional dictionary mapping arguments to numpy dtypes (e.g., {"images": "float32"}).
    """
    def decorator(func: Callable) -> Callable:
        # Production bypass for zero-overhead
        if os.environ.get("TENSORGUARD_ENV") == "production":
            return func
            
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            engine = ShapeEngine()
            import numpy as np
            
            # 1. Validate inputs
            for arg_name, expected_shape in shape_kwargs.items():
                if arg_name in bound_args.arguments:
                    arg_value = bound_args.arguments[arg_name]
                    if not hasattr(arg_value, 'shape'):
                        raise TypeError(f"NumPy Validator: Argument '{arg_name}' must have a '.shape' attribute.")
                    
                    # Ensure it is actually a numpy array (for framework-specific strictness)
                    if not isinstance(arg_value, np.ndarray):
                        raise TypeError(f"NumPy Validator: Argument '{arg_name}' must be a numpy.ndarray, got {type(arg_value).__name__}.")
                        
                    # Dtype check
                    if dtypes is not None and arg_name in dtypes:
                        expected_dtype = dtypes[arg_name]
                        actual_dtype = arg_value.dtype.name
                        if actual_dtype != expected_dtype:
                            raise TypeError(
                                f"NumPy Validator: DType mismatch for '{arg_name}'. "
                                f"Expected '{expected_dtype}', got '{actual_dtype}'."
                            )
                        
                    engine.match_shape(arg_name, arg_value.shape, expected_shape)
                    
            # 2. Execute original function
            result = func(*args, **kwargs)
            
            # 3. Validate return value
            if returns is not None:
                if isinstance(returns, str):
                    if not isinstance(result, np.ndarray):
                        raise TypeError("NumPy Validator: Expected return value to be a numpy.ndarray.")
                    engine.match_shape("return_value", tuple(result.shape), returns)
                elif isinstance(returns, (tuple, list)):
                    if not isinstance(result, (tuple, list)):
                        raise TypeError(f"NumPy Validator: Expected return value to be a Tuple/List, got {type(result).__name__}.")
                    if len(result) != len(returns):
                        raise ValueError(f"NumPy Validator: Expected {len(returns)} return items, got {len(result)}.")
                    for i, (res_tensor, expected_shape) in enumerate(zip(result, returns)):
                        if not isinstance(res_tensor, np.ndarray):
                            raise TypeError(f"NumPy Validator: Return item at index {i} is not a numpy.ndarray.")
                        engine.match_shape(f"return_value[{i}]", tuple(res_tensor.shape), expected_shape)
                elif isinstance(returns, dict):
                    if not isinstance(result, dict):
                        raise TypeError(f"NumPy Validator: Expected return value to be a Dictionary, got {type(result).__name__}.")
                    for key, expected_shape in returns.items():
                        if key not in result:
                            raise KeyError(f"NumPy Validator: Expected key '{key}' not found in returned Dictionary.")
                        res_tensor = result[key]
                        if not isinstance(res_tensor, np.ndarray):
                            raise TypeError(f"NumPy Validator: Return item for key '{key}' is not a numpy.ndarray.")
                        engine.match_shape(f"return_value['{key}']", tuple(res_tensor.shape), expected_shape)
                else:
                    raise TypeError(f"NumPy Validator: Invalid 'returns' argument type. Must be str, tuple, list, or dict.")
                
            return result
            
        return wrapper
    return decorator
