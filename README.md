# batch-lp

[![PyPI version](https://badge.fury.io/py/batch-lp.svg)](https://badge.fury.io/py/batch-lp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance batched linear programming solver using the HiGHS solver with parallel execution capabilities. Built with Rust and optimized for solving multiple LP problems simultaneously.

## Features

- **⚡ Fast Parallel Solving**: Solve multiple LP problems simultaneously using Rayon's work-stealing parallelism
- **🎯 Standard LP Form**: Supports inequality constraints, equality constraints, and variable bounds
- **🐍 Python Bindings**: Easy-to-use Python interface with NumPy arrays via PyO3
- **🚀 High Performance**: Built on the HiGHS solver, one of the fastest open-source LP solvers
- **🔧 Thread Control**: Optionally specify the number of threads for parallel solving

## Installation

### From PyPI (recommended)

```bash
pip install batch-lp
```

### From source

```bash
pip install maturin
git clone https://github.com/landonclark97/batch-lp.git
cd batch-lp
maturin develop --release
```

## Quick Start

### Solve a Single LP

```python
import numpy as np
from batch_lp import solve_lp, Problem

# Problem: min -x - y
#          s.t. x + y <= 2
#               x, y >= 0

c = np.array([-1.0, -1.0])
A = np.array([[1.0, 1.0]])
b = np.array([2.0])

problem = Problem(c=c, A=A, b=b)
solution = solve_lp(problem)

print(f"Status: {solution.status}")
print(f"Objective: {solution.objective}")
print(f"Solution: {solution.x}")
```

### Solve Multiple LPs in Parallel

```python
import numpy as np
from batch_lp import solve_batch_lp, Problem

# Create 100 LP problems using the Problem class
problems = []
for i in range(100):
    problems.append(Problem(
        c=np.array([1.0, -1.0]),
        A=np.array([[1.0, 1.0], [1.0, -1.0]]),
        b=np.array([float(i + 2), 1.0]),
    ))

# Solve all problems in parallel (uses all CPU cores by default)
solutions = solve_batch_lp(problems)

for i, sol in enumerate(solutions[:5]):
    print(f"Problem {i}: {sol.status}, obj={sol.objective}")
```

### Control Number of Threads

```python
# Use 4 threads
solutions = solve_batch_lp(problems, num_threads=4)

# Use single thread (sequential)
solutions = solve_batch_lp(problems, num_threads=1)

# Use all available cores (default)
solutions = solve_batch_lp(problems)
```

## Standard Form

The solver accepts problems in the form:

```
minimize    c^T x
subject to  A x <= b          (inequality constraints)
            A_eq x == b_eq    (equality constraints)
            lb <= x <= ub     (variable bounds)
```

### Problem Class

Both `solve_lp()` and `solve_batch_lp()` use the `Problem` class to define LP problems:

```python
from batch_lp import Problem
import numpy as np

problem = Problem(
    c=np.array([1.0, 2.0]),           # Objective coefficients (required)
    A=np.array([[1.0, 1.0]]),         # Inequality constraint matrix (optional)
    b=np.array([5.0]),                # Inequality RHS (optional)
    A_eq=np.array([[1.0, -1.0]]),     # Equality constraint matrix (optional)
    b_eq=np.array([0.0]),             # Equality RHS (optional)
    bounds=(0, 10),                   # Variable bounds (optional, default: (0, None))
)
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `c` | ndarray | Objective coefficients (n,) | **Required** |
| `A` | ndarray | Inequality constraint matrix (m_ineq, n) | `None` |
| `b` | ndarray | Inequality RHS (m_ineq,) | `None` |
| `A_eq` | ndarray | Equality constraint matrix (m_eq, n) | `None` |
| `b_eq` | ndarray | Equality RHS (m_eq,) | `None` |
| `bounds` | tuple | Variable bounds as `(lower, upper)` | `(0, None)` |

#### Bounds

The `bounds` parameter is a tuple `(lower, upper)` where each element can be:
- `None` — no bound (`-inf` for lower, `+inf` for upper)
- scalar — broadcast to all variables
- ndarray — per-variable bounds

```python
# Default: non-negative variables (0 <= x <= inf)
Problem(c=c, A=A, b=b)

# Scalar bounds broadcast to all variables
Problem(c=c, bounds=(0, 10))

# Completely unbounded variables
Problem(c=c, bounds=(None, None))

# Per-variable bounds
Problem(c=c, bounds=(np.array([0, -1, 0]), np.array([10, 5, 20])))

# Mix scalar and per-variable
Problem(c=c, bounds=(0, np.array([10, 5, 20])))
```

## Examples

See `example.py` for comprehensive examples including:
- Single LP solving
- Equality constraints
- Variable bounds
- Batch solving
- Thread control
- Portfolio optimization

Run the example:

```bash
python example.py
```

## How It Works

**batch-lp** leverages Rust's zero-cost abstractions and Rayon's work-stealing parallelism to efficiently solve multiple LP problems:

1. **No GIL**: Python's Global Interpreter Lock is released during solving, enabling true parallelism
2. **Work Stealing**: Rayon automatically balances load across threads
3. **No Overhead**: Direct Rust ↔ HiGHS communication without Python overhead
4. **Shared Memory**: No data pickling or IPC overhead like multiprocessing

## Development

### Building

```bash
# Development build
maturin develop

# Release build (optimized)
maturin develop --release

# Build wheel
maturin build --release
```

### Testing

```bash
# Run Rust tests
cargo test

# Run Python example
python example.py
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built on top of the excellent [HiGHS](https://highs.dev/) solver
- Uses [Rayon](https://github.com/rayon-rs/rayon) for parallelism
- Python bindings via [PyO3](https://pyo3.rs/)
