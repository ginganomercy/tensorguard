import functools
import inspect
import os
from typing import Any, Callable, Dict, Optional, Union
from tensorguard.core.engine import ShapeEngine

def validate(
    returns: Optional[str] = None, 
    strict_device: bool = True, 
    dtypes: Optional[Dict[str, str]] = None,
    **shape_kwargs
) -> Callable:
    """
    PyTorch specific decorator for validating tensor shapes and devices.
    
    Args:
        returns: Shape string for the return value.
        strict_device: If True, ensures all validated tensors are on the same device.
        dtypes: Optional dictionary mapping argument names to their expected string dtype (e.g., {"images": "float32"}).
    """
    def decorator(func: Callable) -> Callable:
        # Production bypass
        if os.environ.get("TENSORGUARD_ENV") == "production":
            return func
            
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            engine = ShapeEngine()
            import torch
            
            expected_device = None
            
            # 1. Validate inputs
            for arg_name, expected_shape in shape_kwargs.items():
                if arg_name in bound_args.arguments:
                    arg_value = bound_args.arguments[arg_name]
                    
                    if not isinstance(arg_value, torch.Tensor):
                        raise TypeError(f"PyTorch Validator: Argument '{arg_name}' must be a torch.Tensor, got {type(arg_value).__name__}.")
                    
                    # Device consistency check
                    if strict_device:
                        if expected_device is None:
                            expected_device = arg_value.device
                        elif arg_value.device != expected_device:
                            raise RuntimeError(
                                f"PyTorch Validator: Device mismatch! "
                                f"Expected device '{expected_device}', but '{arg_name}' is on '{arg_value.device}'."
                            )
                            
                    # Dtype check
                    if dtypes is not None and arg_name in dtypes:
                        expected_dtype = dtypes[arg_name]
                        actual_dtype = str(arg_value.dtype).replace("torch.", "")
                        if actual_dtype != expected_dtype:
                            raise TypeError(
                                f"PyTorch Validator: DType mismatch for '{arg_name}'. "
                                f"Expected '{expected_dtype}', got '{actual_dtype}'."
                            )
                            
                    engine.match_shape(arg_name, tuple(arg_value.shape), expected_shape)
                    
            # 2. Execute original function
            result = func(*args, **kwargs)
            
            # 3. Validate return value
            if returns is not None:
                if not isinstance(result, torch.Tensor):
                    raise TypeError("PyTorch Validator: Return value must be a torch.Tensor.")
                engine.match_shape("return_value", tuple(result.shape), returns)
                
            return result
            
        return wrapper
    return decorator
