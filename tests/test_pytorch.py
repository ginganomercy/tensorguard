import pytest
import sys
from unittest.mock import MagicMock
from tensorguard.core.engine import ShapeMismatchError
import os

# Mocking PyTorch to avoid a 2GB download in basic testing
mock_torch = MagicMock()

class MockTensor:
    def __init__(self, shape, device="cpu"):
        self.shape = shape
        self.device = device

mock_torch.Tensor = MockTensor
sys.modules['torch'] = mock_torch

from tensorguard.pytorch.decorator import validate

def test_pytorch_validator_basic():
    @validate(images="b c h w", labels="b", returns="b classes")
    def process_data(images, labels):
        batch_size = images.shape[0]
        return MockTensor((batch_size, 10))
        
    img = MockTensor((32, 3, 224, 224))
    lbl = MockTensor((32,))
    
    out = process_data(img, lbl)
    assert out.shape == (32, 10)

def test_pytorch_device_mismatch():
    @validate(images="b c h w", labels="b")
    def process_data(images, labels):
        pass
        
    img = MockTensor((32, 3, 224, 224), device="cuda:0")
    lbl = MockTensor((32,), device="cpu")
    
    with pytest.raises(RuntimeError, match="Device mismatch!"):
        process_data(img, lbl)

def test_pytorch_device_check_disabled():
    @validate(images="b c h w", labels="b", strict_device=False)
    def process_data(images, labels):
        return True
        
    img = MockTensor((32, 3, 224, 224), device="cuda:0")
    lbl = MockTensor((32,), device="cpu")
    
    # Should not raise exception
    assert process_data(img, lbl) is True

def test_pytorch_dtype_mismatch():
    @validate(images="b c", dtypes={"images": "float32"})
    def process_data(images):
        pass
        
    class MockIntTensor(MockTensor):
        def __init__(self):
            self.shape = (32, 3)
            self.dtype = "torch.int64"
            self.device = "cpu"
            
    with pytest.raises(TypeError, match="DType mismatch for 'images'"):
        process_data(MockIntTensor())

def test_pytorch_class_method():
    class DummyModule:
        @validate(images="b c", returns="b 10")
        def forward(self, images, labels=None):
            return MockTensor((images.shape[0], 10))
            
    model = DummyModule()
    img = MockTensor((32, 3))
    
    # Should work perfectly handling 'self'
    out = model.forward(img)
    assert out.shape == (32, 10)

def test_pytorch_tuple_returns():
    @validate(x="b c", returns=("b c", "b 10"))
    def complex_func(x):
        return x, MockTensor((x.shape[0], 10))
    complex_func(MockTensor((32, 3)))
    with pytest.raises(ShapeMismatchError):
        @validate(x="b c", returns=("b c", "b 5"))
        def bad_func(x):
            return x, MockTensor((x.shape[0], 10))
        bad_func(MockTensor((32, 3)))

def test_pytorch_dict_returns():
    @validate(x="b c", returns={"loss": "1", "logits": "b 10"})
    def dict_func(x):
        return {"loss": MockTensor((1,)), "logits": MockTensor((x.shape[0], 10)), "extra": MockTensor((5,))}
    dict_func(MockTensor((32, 3)))
    with pytest.raises(ShapeMismatchError):
        @validate(x="b c", returns={"loss": "2"})
        def bad_dict(x):
            return {"loss": MockTensor((1,))}
        bad_dict(MockTensor((32, 3)))
