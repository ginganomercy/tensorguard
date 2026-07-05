# 🏛️ TensorGuard Architecture

This document serves as an Architectural Decision Record (ADR) and high-level design overview for anyone looking to contribute to the TensorGuard repository.

## 1. Core Philosophy (First-Principles)
TensorGuard is built on the philosophy of **"Declarative Validation with Zero Production Cost."** 
In deep learning engineering, traditional `assert` statements clutter the mathematical domain modeling. TensorGuard shifts this logic entirely to the function signature via Python decorators.

## 2. Why Framework-Specific? (The Adapter Pattern)
Instead of relying on generic duck-typing (e.g. checking `hasattr(tensor, "shape")`), TensorGuard uses the **Ports and Adapters** pattern to create distinct integrations for each ML framework.

**Trade-offs made:**
* **Pros:** We can validate framework-specific paradigms that duck-typing misses. For example, PyTorch's `.device` inconsistency crashes (GPU vs CPU), or TensorFlow's `None` dynamic dimensions in `@tf.function` graph compilation mode.
* **Cons:** Higher maintenance cost due to multiple decorator modules (`tensorguard.pytorch`, `tensorguard.tensorflow`, `tensorguard.numpy`).

## 3. The Validation Pipeline
When a user calls a decorated function during development, TensorGuard executes a 3-step pipeline:

### A. The Parser (`tensorguard.core.parser`)
* Converts the declarative string (`"b c h w"`) into a semantic token list `['b', 'c', 'h', 'w']`.
* **Optimization:** To prevent string manipulation from becoming a CPU bottleneck during tight training loops (e.g., 100k forward passes per epoch), the parser utilizes `@functools.lru_cache`. The parsing overhead is effectively $O(1)$ after the first call.

### B. The Shape Engine (`tensorguard.core.engine`)
* Acts as a state machine during a single function call.
* It remembers variable bindings across arguments. If `images="b c h w"` binds `b=32`, and `labels="b"` evaluates to `16`, the engine instantly triggers a `ShapeMismatchError` using the `rich` library for terminal formatting.

### C. The Decorator Wrapper
* Extracts the Python `inspect.signature` of the original function to accurately map positional and keyword arguments natively.
* Resolves Complex Return Types (Tuples, Lists, Dicts) recursively before submitting their shapes to the Shape Engine.

## 4. Production Bypass
The single most critical requirement of TensorGuard is that it **must not** slow down production inference APIs or hardware deployments.
This is solved via an environment flag `TENSORGUARD_ENV=production`. When enabled, the decorators instantly return the raw, un-wrapped function during import-time, entirely eliminating the wrapper's execution stack and keeping the latency overhead at exactly 0.00ms.
