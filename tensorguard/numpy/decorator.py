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
            
            # 1. Validate inputs
            for arg_name, expected_shape in shape_kwargs.items():
                if arg_name in bound_args.arguments:
                    arg_value = bound_args.arguments[arg_name]
                    if not hasattr(arg_value, 'shape'):
                        raise TypeError(f"NumPy Validator: Argument '{arg_name}' must have a '.shape' attribute.")
                    
                    # Ensure it is actually a numpy array (for framework-specific strictness)
                    import numpy as np
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
                if not hasattr(result, 'shape'):
                    raise TypeError("NumPy Validator: Return value must have a '.shape' attribute.")
                engine.match_shape("return_value", result.shape, returns)
                
            return result
            
        return wrapper
    return decorator
