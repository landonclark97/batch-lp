use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::LpProblem;

/// Extract a single bound element: None → fill with `default`, scalar → broadcast, array → per-variable.
fn extract_bound(_py: Python, obj: &Bound<'_, PyAny>, n: usize, default: f64) -> PyResult<Vec<f64>> {
    if obj.is_none() {
        return Ok(vec![default; n]);
    }
    if let Ok(val) = obj.extract::<f64>() {
        return Ok(vec![val; n]);
    }
    let arr: PyReadonlyArray1<f64> = obj.extract()?;
    Ok(arr.as_slice()?.to_vec())
}

/// Extract bounds tuple: (lower, upper) where each element is None, scalar, or array.
fn extract_bounds(
    py: Python,
    bounds: &PyObject,
    n: usize,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let tuple = bounds.bind(py);
    let tuple = tuple.downcast::<pyo3::types::PyTuple>().map_err(|_| {
        PyValueError::new_err("'bounds' must be a tuple of (lower, upper)")
    })?;
    if tuple.len() != 2 {
        return Err(PyValueError::new_err(
            "'bounds' must be a tuple of exactly 2 elements (lower, upper)",
        ));
    }
    let lb = extract_bound(py, &tuple.get_item(0)?, n, f64::NEG_INFINITY)?;
    let ub = extract_bound(py, &tuple.get_item(1)?, n, f64::INFINITY)?;
    Ok((lb, ub))
}

/// Linear programming problem definition
///
/// Represents an LP problem in standard form:
///     min c^T x
///     s.t. A x <= b
///          A_eq x == b_eq
///          lb <= x <= ub
#[pyclass]
#[allow(non_snake_case)]
pub struct Problem {
    /// Objective coefficients (n,)
    #[pyo3(get, set)]
    pub c: PyObject,

    /// Inequality constraint matrix (m_ineq, n) - optional
    #[pyo3(get, set)]
    #[pyo3(name = "A")]
    pub A: Option<PyObject>,

    /// Inequality RHS (m_ineq,) - optional
    #[pyo3(get, set)]
    pub b: Option<PyObject>,

    /// Equality constraint matrix (m_eq, n) - optional
    #[pyo3(get, set)]
    #[pyo3(name = "A_eq")]
    pub A_eq: Option<PyObject>,

    /// Equality RHS (m_eq,) - optional
    #[pyo3(get, set)]
    pub b_eq: Option<PyObject>,

    /// Variable bounds as (lower, upper) tuple.
    /// Each element: None → no bound (lower=-inf, upper=+inf), scalar → broadcast, array → per-variable.
    #[pyo3(get, set)]
    pub bounds: PyObject,
}

#[pymethods]
impl Problem {
    #[new]
    #[pyo3(signature = (c, A=None, b=None, A_eq=None, b_eq=None, bounds=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        py: Python,
        c: PyObject,
        A: Option<PyObject>,
        b: Option<PyObject>,
        A_eq: Option<PyObject>,
        b_eq: Option<PyObject>,
        bounds: Option<PyObject>,
    ) -> Self {
        let bounds = bounds.unwrap_or_else(|| {
            pyo3::types::PyTuple::new_bound(py, &[0.0.into_py(py), py.None()]).into()
        });
        Problem {
            c,
            A,
            b,
            A_eq,
            b_eq,
            bounds,
        }
    }

    fn __repr__(&self) -> String {
        let mut fields = vec!["c"];
        if self.A.is_some() {
            fields.push("A");
        }
        if self.b.is_some() {
            fields.push("b");
        }
        if self.A_eq.is_some() {
            fields.push("A_eq");
        }
        if self.b_eq.is_some() {
            fields.push("b_eq");
        }
        fields.push("bounds");

        format!("Problem({})", fields.join(", "))
    }
}

/// Python wrapper for LP solution
#[pyclass(name = "Result")]
#[derive(Clone)]
pub struct PySolution {
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub objective: Option<f64>,
    pub x_vec: Option<Vec<f64>>,
}

#[pymethods]
impl PySolution {
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.x_vec {
            Some(x) => Ok(Some(PyArray1::from_vec_bound(py, x.clone()))),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        match self.objective {
            Some(val) => format!("Result(status='{}', objective={:.6})", self.status, val),
            None => format!("Result(status='{}')", self.status),
        }
    }
}

/// Solve a single linear programming problem
///
/// Solves:
///     min c^T x
///     s.t. A x <= b
///          A_eq x == b_eq
///          lb <= x <= ub
///
/// Parameters
/// ----------
/// problem : Problem
///     LP problem instance
///
/// Returns
/// -------
/// Result
///     Result object containing status, objective, and x
#[pyfunction]
#[pyo3(signature = (problem))]
fn solve_lp<'py>(py: Python<'py>, problem: Py<Problem>) -> PyResult<PySolution> {
    let problem_ref = problem.borrow(py);

    // Extract c
    let c_obj = problem_ref.c.bind(py);
    let c: PyReadonlyArray1<f64> = c_obj.extract()?;
    let c_vec = c.as_slice()?.to_vec();
    let n = c_vec.len();

    // Extract A and b
    let (a_ineq, b_ineq) = if let (Some(a_obj), Some(b_obj)) = (&problem_ref.A, &problem_ref.b) {
        let a: PyReadonlyArray2<f64> = a_obj.bind(py).extract()?;
        let b: PyReadonlyArray1<f64> = b_obj.bind(py).extract()?;

        let a_array = a.as_array();
        let b_vec = b.as_slice()?.to_vec();

        let mut a_vec = Vec::new();
        for row_i in 0..a_array.shape()[0] {
            let row: Vec<f64> = (0..a_array.shape()[1])
                .map(|j| a_array[[row_i, j]])
                .collect();
            a_vec.push(row);
        }

        (a_vec, b_vec)
    } else if problem_ref.A.is_some() || problem_ref.b.is_some() {
        return Err(PyValueError::new_err(
            "Both 'A' and 'b' must be provided together",
        ));
    } else {
        (vec![], vec![])
    };

    // Extract A_eq and b_eq
    let (a_eq_vec, b_eq_vec) =
        if let (Some(a_obj), Some(b_obj)) = (&problem_ref.A_eq, &problem_ref.b_eq) {
            let a: PyReadonlyArray2<f64> = a_obj.bind(py).extract()?;
            let b: PyReadonlyArray1<f64> = b_obj.bind(py).extract()?;

            let a_array = a.as_array();
            let b_vec = b.as_slice()?.to_vec();

            let mut a_vec = Vec::new();
            for row_i in 0..a_array.shape()[0] {
                let row: Vec<f64> = (0..a_array.shape()[1])
                    .map(|j| a_array[[row_i, j]])
                    .collect();
                a_vec.push(row);
            }

            (a_vec, b_vec)
        } else if problem_ref.A_eq.is_some() || problem_ref.b_eq.is_some() {
            return Err(PyValueError::new_err(
                "Both 'A_eq' and 'b_eq' must be provided together",
            ));
        } else {
            (vec![], vec![])
        };

    // Extract bounds
    let (lb_vec, ub_vec) = extract_bounds(py, &problem_ref.bounds, n)?;

    // Create and solve problem
    let lp_problem = LpProblem::new(c_vec, a_ineq, b_ineq, a_eq_vec, b_eq_vec, lb_vec, ub_vec)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    py.allow_threads(|| {
        let solution = lp_problem
            .solve()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(PySolution {
            status: solution.status.to_string(),
            objective: solution.objective,
            x_vec: solution.x,
        })
    })
}

/// Solve multiple linear programming problems in parallel
///
/// Parameters
/// ----------
/// problems : list of Problem
///     List of LP problems (Problem instances)
/// num_threads : int, optional
///     Number of threads to use for parallel solving.
///     If None (default), uses all available CPU cores.
///
/// Returns
/// -------
/// list of Result
///     Result for each problem
#[pyfunction]
#[pyo3(signature = (problems, num_threads=None))]
fn solve_batch_lp(
    py: Python,
    problems: Vec<Py<Problem>>,
    num_threads: Option<usize>,
) -> PyResult<Vec<PySolution>> {
    let mut lp_problems = Vec::new();

    // Parse all problems first
    for (i, problem_ref) in problems.iter().enumerate() {
        let problem = problem_ref.borrow(py);

        // Extract c
        let c_obj = problem.c.bind(py);
        let c: PyReadonlyArray1<f64> = c_obj.extract()?;
        let c_vec = c.as_slice()?.to_vec();
        let n = c_vec.len();

        // Extract A and b
        let (a_ineq, b_ineq) = if let (Some(a_obj), Some(b_obj)) = (&problem.A, &problem.b) {
            let a: PyReadonlyArray2<f64> = a_obj.bind(py).extract()?;
            let b: PyReadonlyArray1<f64> = b_obj.bind(py).extract()?;

            let a_array = a.as_array();
            let b_vec = b.as_slice()?.to_vec();

            let mut a_vec = Vec::new();
            for row_i in 0..a_array.shape()[0] {
                let row: Vec<f64> = (0..a_array.shape()[1])
                    .map(|j| a_array[[row_i, j]])
                    .collect();
                a_vec.push(row);
            }

            (a_vec, b_vec)
        } else if problem.A.is_some() || problem.b.is_some() {
            return Err(PyValueError::new_err(format!(
                "Problem {}: Both 'A' and 'b' must be provided together",
                i
            )));
        } else {
            (vec![], vec![])
        };

        // Extract A_eq and b_eq
        let (a_eq_vec, b_eq_vec) =
            if let (Some(a_obj), Some(b_obj)) = (&problem.A_eq, &problem.b_eq) {
                let a: PyReadonlyArray2<f64> = a_obj.bind(py).extract()?;
                let b: PyReadonlyArray1<f64> = b_obj.bind(py).extract()?;

                let a_array = a.as_array();
                let b_vec = b.as_slice()?.to_vec();

                let mut a_vec = Vec::new();
                for row_i in 0..a_array.shape()[0] {
                    let row: Vec<f64> = (0..a_array.shape()[1])
                        .map(|j| a_array[[row_i, j]])
                        .collect();
                    a_vec.push(row);
                }

                (a_vec, b_vec)
            } else if problem.A_eq.is_some() || problem.b_eq.is_some() {
                return Err(PyValueError::new_err(format!(
                    "Problem {}: Both 'A_eq' and 'b_eq' must be provided together",
                    i
                )));
            } else {
                (vec![], vec![])
            };

        // Extract bounds
        let (lb_vec, ub_vec) = extract_bounds(py, &problem.bounds, n)?;

        let lp_problem = LpProblem::new(c_vec, a_ineq, b_ineq, a_eq_vec, b_eq_vec, lb_vec, ub_vec)
            .map_err(|e| PyValueError::new_err(format!("Problem {}: {}", i, e)))?;

        lp_problems.push(lp_problem);
    }

    // Solve all problems in parallel (releasing GIL)
    let solutions = py.allow_threads(|| match num_threads {
        Some(n) => crate::solve_batch_with_threads(&lp_problems, n),
        None => crate::solve_batch(&lp_problems),
    });

    // Convert to Python solutions
    solutions
        .into_iter()
        .enumerate()
        .map(|(i, result)| {
            result
                .map(|sol| PySolution {
                    status: sol.status.to_string(),
                    objective: sol.objective,
                    x_vec: sol.x,
                })
                .map_err(|e| PyRuntimeError::new_err(format!("Problem {}: {}", i, e)))
        })
        .collect()
}

/// Python module for batched linear programming
#[pymodule]
fn batch_lp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_lp, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch_lp, m)?)?;
    m.add_class::<Problem>()?;
    m.add_class::<PySolution>()?;
    Ok(())
}
