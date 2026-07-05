import re
from typing import List, Union

def parse_shape_string(shape_str: str) -> List[Union[str, int]]:
    """
    Parses a declarative shape string into a list of semantic tokens.
    
    Examples:
        "b c h w" -> ['b', 'c', 'h', 'w']
        "32 3 224 224" -> [32, 3, 224, 224]
        "b, c, h, w" -> ['b', 'c', 'h', 'w']
        "..." -> ['...']
        "b ... w" -> ['b', '...', 'w']
    """
    if not isinstance(shape_str, str):
        raise TypeError(f"Shape must be a string, got {type(shape_str).__name__}")
    
    # Normalize separators (comma to space) and split by whitespace
    normalized = shape_str.replace(",", " ").strip()
    if not normalized:
        return []
        
    raw_tokens = normalized.split()
    tokens = []
    
    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue
            
        # Check if it's an exact dimension (integer)
        if token.isdigit():
            tokens.append(int(token))
        # Check if it's a valid variable name or wildcard (...)
        elif token == "..." or token == "*":
            tokens.append("...")
        elif re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", token):
            tokens.append(token)
        else:
            raise ValueError(f"Invalid dimension token: '{token}' in shape '{shape_str}'. "
                             "Tokens must be alphanumeric variable names, integers, or '...'")
                             
    return tokens
