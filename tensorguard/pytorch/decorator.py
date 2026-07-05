import functools
import inspect
import os
from typing import Any, Callable, Dict, Optional, Union, Tuple, List
from tensorguard.core.engine import ShapeEngine

def validate(
    returns: Optional[Union[str, Tuple[str, ...], List[str], Dict[str, str]]] = None, 
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
                if isinstance(returns, str):
                    if not isinstance(result, torch.Tensor):
                        raise TypeError("PyTorch Validator: Expected return value to be a single torch.Tensor.")
                    engine.match_shape("return_value", tuple(result.shape), returns)
                elif isinstance(returns, (tuple, list)):
                    if not isinstance(result, (tuple, list)):
                        raise TypeError(f"PyTorch Validator: Expected return value to be a Tuple/List, got {type(result).__name__}.")
                    if len(result) != len(returns):
                        raise ValueError(f"PyTorch Validator: Expected {len(returns)} return items, got {len(result)}.")
                    for i, (res_tensor, expected_shape) in enumerate(zip(result, returns)):
                        if not isinstance(res_tensor, torch.Tensor):
                            raise TypeError(f"PyTorch Validator: Return item at index {i} is not a torch.Tensor.")
                        engine.match_shape(f"return_value[{i}]", tuple(res_tensor.shape), expected_shape)
                elif isinstance(returns, dict):
                    if not isinstance(result, dict):
                        raise TypeError(f"PyTorch Validator: Expected return value to be a Dictionary, got {type(result).__name__}.")
                    for key, expected_shape in returns.items():
                        if key not in result:
                            raise KeyError(f"PyTorch Validator: Expected key '{key}' not found in returned Dictionary.")
                        res_tensor = result[key]
                        if not isinstance(res_tensor, torch.Tensor):
                            raise TypeError(f"PyTorch Validator: Return item for key '{key}' is not a torch.Tensor.")
                        engine.match_shape(f"return_value['{key}']", tuple(res_tensor.shape), expected_shape)
                else:
                    raise TypeError(f"PyTorch Validator: Invalid 'returns' argument type. Must be str, tuple, list, or dict.")
                
            return result
            
        return wrapper
    return decorator
