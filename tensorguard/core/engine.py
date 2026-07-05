from typing import List, Union, Tuple, Dict, Any
from .parser import parse_shape_string
from .exceptions import ShapeMismatchError

class ShapeEngine:
    def __init__(self) -> None:
        # Stores the resolved sizes for dynamic variables across the entire validation context
        self.context: Dict[str, int] = {}
        
    def match_shape(self, tensor_name: str, actual_shape: Tuple[int, ...], expected_shape_str: str) -> None:
        """
        Validates if the actual_shape matches the parsed expected_shape_str.
        It also updates the global context to ensure consistency between different tensors.
        """
        tokens = parse_shape_string(expected_shape_str)
        
        # Handle wildcard matching (e.g., "b ... w")
        has_wildcard = "..." in tokens
        
        if not has_wildcard and len(actual_shape) != len(tokens):
            raise ShapeMismatchError(
                tensor_name=tensor_name,
                expected_shape=expected_shape_str,
                actual_shape=actual_shape,
                error_msg=f"Expected {len(tokens)} dimensions, but got {len(actual_shape)}."
            )
            
        if has_wildcard:
            # We only support one wildcard per shape string for simplicity in MVP
            wildcard_count = tokens.count("...")
            if wildcard_count > 1:
                raise ValueError("Only one wildcard '...' is allowed per shape definition.")
            
            wildcard_idx = tokens.index("...")
            left_tokens = tokens[:wildcard_idx]
            right_tokens = tokens[wildcard_idx+1:]
            
            if len(actual_shape) < len(left_tokens) + len(right_tokens):
                raise ShapeMismatchError(
                    tensor_name=tensor_name,
                    expected_shape=expected_shape_str,
                    actual_shape=actual_shape,
                    error_msg="Tensor is too small to match the required dimensions around wildcard."
                )
            
            # Reconstruct the tokens sequence replacing the wildcard with dummy tokens
            num_wildcard_dims = len(actual_shape) - len(left_tokens) - len(right_tokens)
            wildcard_fill = ["*"] * num_wildcard_dims
            tokens = left_tokens + wildcard_fill + right_tokens

        # Iterate and match each dimension
        for dim_idx, (token, actual_dim_size) in enumerate(zip(tokens, actual_shape)):
            if token == "*" or actual_dim_size == -1:
                continue  # Skip wildcard filled dimensions or symbolic (unknown) dimensions
                
            if isinstance(token, int):
                if actual_dim_size != token:
                    raise ShapeMismatchError(
                        tensor_name=tensor_name,
                        expected_shape=expected_shape_str,
                        actual_shape=actual_shape,
                        error_msg=f"Dimension {dim_idx} mismatch. Expected exactly {token}, got {actual_dim_size}."
                    )
            elif isinstance(token, str):
                if token in self.context:
                    # The variable has been seen before, check for consistency
                    if self.context[token] != actual_dim_size:
                        raise ShapeMismatchError(
                            tensor_name=tensor_name,
                            expected_shape=expected_shape_str,
                            actual_shape=actual_shape,
                            error_msg=(
                                f"Inconsistent size for variable '{token}'. "
                                f"Previously resolved as {self.context[token]}, "
                                f"but currently got {actual_dim_size}."
                            )
                        )
                else:
                    # First time seeing this variable, store it in context
                    self.context[token] = actual_dim_size
