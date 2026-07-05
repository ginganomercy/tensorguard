import pytest
from tensorguard.core.parser import parse_shape_string
from tensorguard.core.engine import ShapeEngine, ShapeMismatchError

def test_parse_shape_string():
    assert parse_shape_string("b c h w") == ['b', 'c', 'h', 'w']
    assert parse_shape_string("32 3 224 224") == [32, 3, 224, 224]
    assert parse_shape_string("b, c, h, w") == ['b', 'c', 'h', 'w']
    assert parse_shape_string("...") == ['...']
    assert parse_shape_string("b ... w") == ['b', '...', 'w']
    
    with pytest.raises(ValueError):
        parse_shape_string("b c h w @")

def test_shape_engine_basic_match():
    engine = ShapeEngine()
    # Mocking actual tensor shape as a tuple
    engine.match_shape("images", (32, 3, 224, 224), "b c h w")
    
    assert engine.context["b"] == 32
    assert engine.context["c"] == 3
    assert engine.context["h"] == 224
    assert engine.context["w"] == 224

def test_shape_engine_mismatch():
    engine = ShapeEngine()
    
    with pytest.raises(ShapeMismatchError, match="Expected 4 dimensions"):
        engine.match_shape("images", (32, 3, 224), "b c h w")
        
def test_shape_engine_consistency():
    engine = ShapeEngine()
    engine.match_shape("images", (32, 3, 224, 224), "b c h w")
    
    # Matching labels with a consistent batch size (32)
    engine.match_shape("labels", (32,), "b")
    
    # Matching another tensor with an inconsistent batch size (16)
    with pytest.raises(ShapeMismatchError, match="Inconsistent size for variable"):
        engine.match_shape("masks", (16, 224, 224), "b h w")

def test_shape_engine_exact_dimensions():
    engine = ShapeEngine()
    # Matching exact integers
    engine.match_shape("images", (32, 3, 224, 224), "b 3 h w")
    
    with pytest.raises(ShapeMismatchError, match="Expected exactly 3"):
        engine.match_shape("images2", (32, 1, 224, 224), "b 3 h w")

def test_shape_engine_wildcards():
    engine = ShapeEngine()
    engine.match_shape("images", (32, 3, 224, 224), "b ... w")
    assert engine.context["b"] == 32
    assert engine.context["w"] == 224
    
    with pytest.raises(ShapeMismatchError):
        engine.match_shape("too_small", (32,), "b ... w")
