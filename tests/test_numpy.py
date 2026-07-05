import pytest
import numpy as np
from tensorguard.numpy.decorator import validate
from tensorguard.core.engine import ShapeMismatchError
import os

def test_numpy_validator_basic():
    @validate(images="b c h w", labels="b", returns="b classes")
    def process_data(images, labels):
        # simulate returning predictions (batch_size, 10)
        batch_size = images.shape[0]
        return np.zeros((batch_size, 10))
        
    img = np.random.randn(32, 3, 224, 224)
    lbl = np.random.randint(0, 10, size=(32,))
    
    # Should pass without error
    out = process_data(img, lbl)
    assert out.shape == (32, 10)

def test_numpy_validator_mismatch():
    @validate(images="b c h w", labels="b")
    def process_data(images, labels):
        pass
        
    img = np.random.randn(32, 3, 224, 224)
    lbl = np.random.randint(0, 10, size=(16,)) # Mismatched batch size
    
    with pytest.raises(ShapeMismatchError, match="Inconsistent size for variable 'b'"):
        process_data(img, lbl)
        
def test_numpy_validator_type_strictness():
    @validate(images="b c")
    def process_data(images):
        pass
        
    class FakeTensor:
        shape = (32, 64)
        
    # FakeTensor has shape, but is not np.ndarray.
    with pytest.raises(TypeError, match="must be a numpy.ndarray"):
        process_data(FakeTensor())

def test_numpy_dtype_mismatch():
    @validate(images="b c", dtypes={"images": "float32"})
    def process_data(images):
        pass
        
    # Create an int64 array instead of float32
    img = np.zeros((32, 64), dtype=np.int64)
    with pytest.raises(TypeError, match="DType mismatch for 'images'"):
        process_data(img)

def test_numpy_validator_production_bypass():
    os.environ["TENSORGUARD_ENV"] = "production"
    
    @validate(images="b 3 h w")
    def process_data(images):
        return True
        
    # Should completely bypass and not throw a shape error
    wrong_shape_img = np.random.randn(32, 1, 224, 224)
    assert process_data(wrong_shape_img) is True
    
    # Clean up
    del os.environ["TENSORGUARD_ENV"]
