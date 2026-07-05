import pytest
import sys
from unittest.mock import MagicMock
from tensorguard.core.engine import ShapeMismatchError
import os

mock_tf = MagicMock()

class MockTensorShape:
    def __init__(self, shape_list):
        self._shape = shape_list
        
    def as_list(self):
        return self._shape

class MockTensor:
    def __init__(self, shape_list):
        self.shape = MockTensorShape(shape_list)

mock_tf.Tensor = MockTensor
sys.modules['tensorflow'] = mock_tf

from tensorguard.tensorflow.decorator import validate

def test_tensorflow_validator_basic():
    @validate(images="b c h w", labels="b", returns="b classes")
    def process_data(images, labels):
        return MockTensor([32, 10])
        
    img = MockTensor([32, 3, 224, 224])
    lbl = MockTensor([32])
    
    out = process_data(img, lbl)
    assert out.shape.as_list() == [32, 10]

def test_tensorflow_dynamic_dimensions():
    # In graph mode, batch size might be None
    @validate(images="b 3 224 224", returns="b 10")
    def process_data(images):
        return MockTensor([None, 10])
        
    img = MockTensor([None, 3, 224, 224])
    
    # This should pass without raising an error because -1 skips validation
    out = process_data(img)
    assert out.shape.as_list() == [None, 10]

def test_tensorflow_shape_mismatch():
    @validate(images="b 3 224 224")
    def process_data(images):
        pass
        
    img = MockTensor([32, 1, 224, 224])
    
    with pytest.raises(ShapeMismatchError, match="Dimension 1 mismatch"):
        process_data(img)

def test_tensorflow_dtype_mismatch():
    @validate(images="b 3", dtypes={"images": "float32"})
    def process_data(images):
        pass
        
    class MockIntTensor(MockTensor):
        def __init__(self):
            self.shape = MockTensorShape([32, 3])
            class MockDtype:
                name = "int32"
            self.dtype = MockDtype()
            
    with pytest.raises(TypeError, match="DType mismatch for 'images'"):
        process_data(MockIntTensor())
