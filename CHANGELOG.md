# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2026-02-16

### Added
- Support for SciPy sparse matrices as `A` and `A_eq` inputs (csr_matrix, csr_array, coo_matrix, etc.)
- Sparse data is extracted directly from CSR arrays without densification

### Changed
- Internal constraint representation changed from dense rows to sparse rows `(col_index, coeff)`
- Deduplicated matrix extraction logic in Python bindings

## [1.0.2] - 2026-02-11

### Changed
- Replaced `lb` and `ub` parameters with a single `bounds` tuple parameter `(lower, upper)`
- Each element can be `None` (default), a scalar (broadcast), or an array (per-variable)
- Default is `(0, None)` which gives `0 <= x <= inf` (same behavior as before)

## [1.0.1] - 2026-02-11

### Added
- Initial release
- Single LP solving with `solve_lp()`
- Batch parallel LP solving with `solve_batch_lp()`
- `Problem` dataclass for defining LP problems (cleaner API than dictionaries)
- Support for inequality constraints (A x <= b)
- Support for equality constraints (A_eq x == b_eq)
- Support for variable bounds (lb <= x <= ub)
- Thread control via `num_threads` parameter
- Python bindings via PyO3
- Rayon-based work-stealing parallelism
- Comprehensive examples in `example.py`
- Performance benchmarks showing 10-15x speedup over scipy
- Automated CI/CD with GitHub Actions
- Automated PyPI publishing on release

### Technical Details
- NumPy >= 2.0 requirement
- Multi-platform support (Linux, macOS, Windows)
- Python 3.9+ support
- Zero GIL overhead for true parallelism

[1.0.3]: https://github.com/landonclark97/batch-lp/releases/tag/v1.0.3
[1.0.2]: https://github.com/landonclark97/batch-lp/releases/tag/v1.0.2
[1.0.1]: https://github.com/landonclark97/batch-lp/releases/tag/v1.0.1
