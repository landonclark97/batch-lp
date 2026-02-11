use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::LpProblem;

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

    /// Lower bounds (n,) - optional, defaults to zeros
    #[pyo3(get, set)]
    pub lb: Option<PyObject>,

    /// Upper bounds (n,) - optional, defaults to infinity
    #[pyo3(get, set)]
    pub ub: Option<PyObject>,
}

#[pymethods]
impl Problem {
    #[new]
    #[pyo3(signature = (c, A=None, b=None, A_eq=None, b_eq=None, lb=None, ub=None))]
    #[allow(non_snake_case)]
    fn new(
        c: PyObject,
        A: Option<PyObject>,
        b: Option<PyObject>,
        A_eq: Option<PyObject>,
        b_eq: Option<PyObject>,
        lb: Option<PyObject>,
        ub: Option<PyObject>,
    ) -> Self {
        Problem {
            c,
            A,
            b,
            A_eq,
            b_eq,
            lb,
            ub,
        }
    }

    fn __repr__(&self) -> String {
        let fields = vec![
            "c",
            if self.A.is_some() { "A" } else { "" },
            if self.b.is_some() { "b" } else { "" },
            if self.A_eq.is_some() { "A_eq" } else { "" },
            if self.b_eq.is_some() { "b_eq" } else { "" },
            if self.lb.is_some() { "lb" } else { "" },
            if self.ub.is_some() { "ub" } else { "" },
        ]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(", ");

        format!("Problem({})", fields)
    }
}

/// Python wrapper for LP solution
#[pyclass]
#[derive(Clone)]
pub struct PySolution {
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub objective_value: Option<f64>,
    pub x: Option<Vec<f64>>,
}

#[pymethods]
impl PySolution {
    /// Get the solution vector as a numpy array
    fn get_x<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.x {
            Some(x) => Ok(Some(PyArray1::from_vec_bound(py, x.clone()))),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        match self.objective_value {
            Some(val) => format!("Solution(status='{}', objective={:.6})", self.status, val),
            None => format!("Solution(status='{}')", self.status),
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
/// PySolution
///     Solution object containing status, objective_value, and x
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
    let lb_vec = if let Some(lb_obj) = &problem_ref.lb {
        let lb: PyReadonlyArray1<f64> = lb_obj.bind(py).extract()?;
        lb.as_slice()?.to_vec()
    } else {
        vec![0.0; n]
    };

    let ub_vec = if let Some(ub_obj) = &problem_ref.ub {
        let ub: PyReadonlyArray1<f64> = ub_obj.bind(py).extract()?;
        ub.as_slice()?.to_vec()
    } else {
        vec![f64::INFINITY; n]
    };

    // Create and solve problem
    let lp_problem = LpProblem::new(c_vec, a_ineq, b_ineq, a_eq_vec, b_eq_vec, lb_vec, ub_vec)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    py.allow_threads(|| {
        let solution = lp_problem
            .solve()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(PySolution {
            status: solution.status.to_string(),
            objective_value: solution.objective_value,
            x: solution.x,
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
/// list of PySolution
///     Solutions for each problem
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
        let lb_vec = if let Some(lb_obj) = &problem.lb {
            let lb: PyReadonlyArray1<f64> = lb_obj.bind(py).extract()?;
            lb.as_slice()?.to_vec()
        } else {
            vec![0.0; n]
        };

        let ub_vec = if let Some(ub_obj) = &problem.ub {
            let ub: PyReadonlyArray1<f64> = ub_obj.bind(py).extract()?;
            ub.as_slice()?.to_vec()
        } else {
            vec![f64::INFINITY; n]
        };

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
                    objective_value: sol.objective_value,
                    x: sol.x,
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
