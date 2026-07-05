<div align="center">
  <h1>TensorGuard</h1>
  <p><strong>A zero-overhead, declarative tensor shape and dtype validator for PyTorch, TensorFlow, and NumPy.</strong></p>
  
  ![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
  ![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
  ![License](https://img.shields.io/badge/License-MIT-green)
</div>

---

## The Problem: Silent Broadcasting & Shape Mismatches

If you are a Data Scientist, ML Researcher, or AI Engineer, you know the pain: you wait 4 hours for a model to train, only for it to crash due to a **Tensor Shape Mismatch** deep inside a loss function. Even worse, **Silent Broadcasting** might occur where PyTorch or TensorFlow silently reshapes your matrices, leading to entirely incorrect mathematical calculations without throwing a single error.

Standard validation using `assert x.shape == (batch, channels, h, w)` has major flaws:
1. It is repetitive and clutters your clean domain logic.
2. It evaluates every time, slowing down your high-performance training loops.
3. It cannot easily track variable consistencies (e.g., ensuring `batch_size` matches across 3 different arguments).
4. Error traces are unreadable.

## The Solution: TensorGuard

**TensorGuard** introduces a beautiful, declarative decorator approach to validate tensor structures at the boundaries of your functions. Built from First-Principles, it acts as a strict compile-time/runtime checker during development, and completely vanishes during production.

### Key Features
* **Zero-Overhead in Production**: Set `TENSORGUARD_ENV=production` and the decorators instantly return the original function, adding **0 nanoseconds** of latency to your production pipelines.
* **Cross-Argument Consistency**: If you declare `images="b c h w"` and `labels="b"`, TensorGuard guarantees the value of `b` is mathematically identical for both.
* **Rich Error Formatting**: When a mismatch occurs, TensorGuard prints a beautiful, colorful visual table pointing exactly to which dimension failed, drastically reducing debugging time.
* **Framework-Specific Strictness**:
  * `tensorguard.pytorch`: Automatically enforces `.device` consistency (no more `cuda:0` vs `cpu` crashes) and checks `torch.dtype`.
  * `tensorguard.tensorflow`: Gracefully handles `None` dynamic dimensions in `@tf.function` graph compilation mode.
  * `tensorguard.numpy`: Ensures strict adherence to `np.ndarray` structures for data engineering pipelines.

---

## Installation

Install TensorGuard via pip:
```bash
pip install tensor-shape-guard
```

---

## Quick Start & Usage

### 1. PyTorch 
TensorGuard shines in PyTorch by also automatically validating `device` alignment.

```python
import torch
from tensorguard.pytorch import validate

class VisionModel(torch.nn.Module):
    # Enforces shape logic, return shapes, and dtypes
    @validate(
        images="batch channels height width", 
        labels="batch", 
        dtypes={"images": "float32"},
        returns="batch classes"
    )
    def forward(self, images, labels):
        # Business logic goes here
        # If images is on cuda:0 and labels is on cpu, TensorGuard catches it!
        return torch.zeros((images.shape[0], 10))

model = VisionModel()
img = torch.randn(32, 3, 224, 224, dtype=torch.float32)
lbl = torch.randint(0, 10, (32,))

# Works flawlessly
model(img, lbl) 
```

### 2. TensorFlow
TensorGuard handles Symbolic variables seamlessly.

```python
import tensorflow as tf
from tensorguard.tensorflow import validate

# Handles 'None' dynamic dimensions gracefully in graph mode
@validate(images="batch 224 224 3", dtypes={"images": "float32"})
@tf.function
def model_step(images):
    return tf.zeros((tf.shape(images)[0], 10))
```

---

## Complex Return Validation (Tuple & Dict) [New in v0.2.0]

TensorGuard natively understands complex data structures like `Tuple` and `Dict`, which are heavily used in modern architectures like HuggingFace Transformers or GANs. The syntax perfectly mirrors the Python data structures you already use, making it incredibly intuitive and human-readable compared to manual `assert` statements.

**Tuple Validation:**
```python
@validate(images="b c h w", returns=("b c h w", "b 10"))
def forward(images):
    # Returns a tuple of (reconstructed_image, logits)
    return torch.zeros_like(images), torch.zeros(images.shape[0], 10)
```

**Dictionary Validation (Partial Matching):**
```python
@validate(input_ids="b seq", returns={"loss": "1", "hidden_states": "b seq 768"})
def forward(input_ids):
    return {
        "loss": torch.tensor(0.5),
        "hidden_states": torch.randn(input_ids.shape[0], input_ids.shape[1], 768),
        "attentions": torch.randn(4, 4) # Ignored by TensorGuard (Flexible Partial Validation)
    }
```

---

## How to Bypass in Production
TensorGuard is built for zero-compromise performance. Once you are confident your shapes are correct and you are deploying to an edge device or high-traffic API, simply set the environment variable:

```bash
export TENSORGUARD_ENV=production
```
*(All `@validate` decorators will now have zero overhead).*

---

## Support This Project

TensorGuard is an open-source project. If it has saved you valuable GPU hours and debugging time, consider supporting the creator by following on Instagram!

[![Follow on Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/galaxy_scream)

---

## Contributing & Testing

We welcome PRs! To run the test suite locally:
```bash
# Clone the repository
git clone https://github.com/ginganomercy/tensorguard.git
cd tensorguard

# Install with development dependencies
pip install -e .[dev,numpy]

# Run tests
pytest tests/
```
