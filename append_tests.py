with open('tests/test_pytorch.py', 'a', encoding='utf-8') as f:
    f.write('''
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
''')

with open('tests/test_tensorflow.py', 'a', encoding='utf-8') as f:
    f.write('''
def test_tensorflow_tuple_returns():
    @validate(x="b c", returns=("b c", "b 10"))
    def complex_func(x):
        # We need MockTensor which takes a shape_list
        # x is MockTensor, so we use its internal _shape from MockTensorShape
        return x, MockTensor([x.shape.as_list()[0], 10])
    complex_func(MockTensor([32, 3]))
    with pytest.raises(ShapeMismatchError):
        @validate(x="b c", returns=("b c", "b 5"))
        def bad_func(x):
            return x, MockTensor([x.shape.as_list()[0], 10])
        bad_func(MockTensor([32, 3]))

def test_tensorflow_dict_returns():
    @validate(x="b c", returns={"loss": "1", "logits": "b 10"})
    def dict_func(x):
        return {"loss": MockTensor([1]), "logits": MockTensor([x.shape.as_list()[0], 10]), "extra": MockTensor([5])}
    dict_func(MockTensor([32, 3]))
    with pytest.raises(ShapeMismatchError):
        @validate(x="b c", returns={"loss": "2"})
        def bad_dict(x):
            return {"loss": MockTensor([1])}
        bad_dict(MockTensor([32, 3]))
''')

with open('tests/test_numpy.py', 'a', encoding='utf-8') as f:
    f.write('''
def test_numpy_tuple_returns():
    @validate(x="b c", returns=("b c", "b 10"))
    def complex_func(x):
        return x, np.zeros((x.shape[0], 10))
    complex_func(np.zeros((32, 3)))
    with pytest.raises(ShapeMismatchError):
        @validate(x="b c", returns=("b c", "b 5"))
        def bad_func(x):
            return x, np.zeros((x.shape[0], 10))
        bad_func(np.zeros((32, 3)))

def test_numpy_dict_returns():
    @validate(x="b c", returns={"loss": "1", "logits": "b 10"})
    def dict_func(x):
        return {"loss": np.zeros(1), "logits": np.zeros((x.shape[0], 10)), "extra": np.zeros(5)}
    dict_func(np.zeros((32, 3)))
    with pytest.raises(ShapeMismatchError):
        @validate(x="b c", returns={"loss": "2"})
        def bad_dict(x):
            return {"loss": np.zeros(1)}
        bad_dict(np.zeros((32, 3)))
''')
