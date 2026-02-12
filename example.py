#!/usr/bin/env python3
"""
Example usage of batch-lp library for solving linear programs.

This demonstrates both single LP solving and batched parallel solving.
"""

import numpy as np
import time
from batch_lp import solve_lp, solve_batch_lp, Problem


def example_single_lp():
    """
    Solve a single LP:
        min  -x - y
        s.t.  x + y <= 2
              x, y >= 0

    Expected solution: x=2, y=0 (or x=0, y=2), obj=-2
    """
    print("=" * 60)
    print("Example 1: Single LP")
    print("=" * 60)

    c = np.array([-1.0, -1.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([2.0])

    problem = Problem(c=c, A=A, b=b)
    solution = solve_lp(problem)

    print(f"Status: {solution.status}")
    print(f"Objective value: {solution.objective}")
    print(f"Solution: {solution.x}")
    print()


def example_with_equality():
    """
    Solve an LP with equality constraints:
        min   x + 2y
        s.t.  x + y == 3
              x, y >= 0

    Expected solution: x=3, y=0, obj=3
    """
    print("=" * 60)
    print("Example 2: LP with Equality Constraints")
    print("=" * 60)

    c = np.array([1.0, 2.0])
    A_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([3.0])

    problem = Problem(c=c, A_eq=A_eq, b_eq=b_eq)
    solution = solve_lp(problem)

    print(f"Status: {solution.status}")
    print(f"Objective value: {solution.objective}")
    print(f"Solution: {solution.x}")
    print()


def example_with_bounds():
    """
    Solve an LP with variable bounds:
        min   x + y
        s.t.  x + 2y >= 4
              1 <= x <= 5
              2 <= y <= 6

    Note: We convert >= to <= by multiplying by -1
    """
    print("=" * 60)
    print("Example 3: LP with Variable Bounds")
    print("=" * 60)

    c = np.array([1.0, 1.0])
    A = np.array([[-1.0, -2.0]])  # -(x + 2y) <= -4  =>  x + 2y >= 4
    b = np.array([-4.0])
    lb = np.array([1.0, 2.0])
    ub = np.array([5.0, 6.0])

    problem = Problem(c=c, A=A, b=b, lb=lb, ub=ub)
    solution = solve_lp(problem)

    print(f"Status: {solution.status}")
    print(f"Objective value: {solution.objective}")
    print(f"Solution: {solution.x}")
    print()


def example_batch_solve():
    """
    Solve multiple LPs in parallel.
    """
    print("=" * 60)
    print("Example 4: Batch Solving (Parallel)")
    print("=" * 60)

    # Create 100 similar LP problems with different parameters using Problem class
    problems = []
    for i in range(100):
        problems.append(Problem(
            c=np.array([1.0, -1.0]),
            A=np.array([[1.0, 1.0], [1.0, -1.0]]),
            b=np.array([float(i + 2), 1.0]),
        ))

    print(f"Solving {len(problems)} LPs in parallel...")
    start = time.time()
    solutions = solve_batch_lp(problems)
    elapsed = time.time() - start

    print(f"Solved in {elapsed:.4f} seconds")
    print(f"Average time per LP: {elapsed/len(problems)*1000:.3f} ms\n")

    # Show first few results
    for i, sol in enumerate(solutions[:5]):
        print(f"Problem {i}: status={sol.status}, objective={sol.objective:.4f}")
    print()


def example_thread_control():
    """
    Demonstrate controlling the number of threads for parallel solving.
    """
    print("=" * 60)
    print("Example 5: Thread Control")
    print("=" * 60)

    # Create 50 random LPs using Problem class
    problems = []
    for i in range(50):
        n_vars = 20
        n_constraints = 40
        problems.append(Problem(
            c=np.random.randn(n_vars),
            A=np.random.randn(n_constraints, n_vars),
            b=np.random.rand(n_constraints) * 10,
            lb=np.zeros(n_vars),
            ub=np.ones(n_vars) * 10,
        ))

    # Solve with different thread counts
    for num_threads in [1, 4, None]:  # None = auto (all cores)
        start = time.time()
        solutions = solve_batch_lp(problems, num_threads=num_threads)
        elapsed = time.time() - start

        label = f"{num_threads} thread(s)" if num_threads else "all cores"
        print(f"Using {label:12}: {elapsed:.4f}s")

    print()


def example_portfolio_optimization():
    """
    Example: Simple portfolio optimization

    Maximize expected return subject to:
    - Sum of weights = 1
    - 0 <= weight <= 0.4 (max 40% in any asset)
    """
    print("=" * 60)
    print("Example 6: Portfolio Optimization")
    print("=" * 60)

    n_assets = 5

    # Expected returns for 5 assets
    expected_returns = np.array([0.05, 0.08, 0.12, 0.07, 0.10])
    c = -expected_returns  # Maximize by minimizing negative

    # Constraint: Sum of weights = 1
    A_eq = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    b_eq = np.array([1.0])

    # Bounds: 0 <= weight <= 0.4 (max 40% in any asset)
    lb = np.zeros(n_assets)
    ub = np.full(n_assets, 0.4)

    problem = Problem(c=c, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub)
    solution = solve_lp(problem)

    print(f"Status: {solution.status}")
    print(f"Expected return: {-solution.objective:.4f}")
    print(f"Optimal weights: {solution.x}")
    print(f"Sum of weights: {np.sum(solution.x):.6f}")
    print()


def main():
    """Run all examples."""
    print("\nHiGHS Batch Linear Programming Solver - Examples\n")

    example_single_lp()
    example_with_equality()
    example_with_bounds()
    example_batch_solve()
    example_thread_control()
    example_portfolio_optimization()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
