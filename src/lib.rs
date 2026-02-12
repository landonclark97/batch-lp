use highs::{Col, RowProblem, Sense};
use rayon::prelude::*;
use std::fmt;

#[allow(clippy::useless_conversion, clippy::needless_lifetimes)]
pub mod python;

/// Error types for the batched LP solver
#[derive(Debug, thiserror::Error)]
pub enum BatchLpError {
    #[error("Invalid problem dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Solver error: {0}")]
    SolverError(String),

    #[error("No solution found")]
    NoSolution,
}

pub type Result<T> = std::result::Result<T, BatchLpError>;

/// Status of an LP solution
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolutionStatus {
    Optimal,
    Infeasible,
    Unbounded,
    Error,
}

impl fmt::Display for SolutionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolutionStatus::Optimal => write!(f, "Optimal"),
            SolutionStatus::Infeasible => write!(f, "Infeasible"),
            SolutionStatus::Unbounded => write!(f, "Unbounded"),
            SolutionStatus::Error => write!(f, "Error"),
        }
    }
}

/// Result of solving a single LP
#[derive(Debug, Clone)]
pub struct LpSolution {
    pub status: SolutionStatus,
    pub objective: Option<f64>,
    pub x: Option<Vec<f64>>,
}

/// Linear programming problem in standard form:
/// min c^T x
/// s.t. A x <= b
///      A_eq x == b_eq
///      lb <= x <= ub
#[derive(Debug, Clone)]
pub struct LpProblem {
    pub c: Vec<f64>,           // Objective coefficients (n,)
    pub a_ineq: Vec<Vec<f64>>, // Inequality constraint matrix (m_ineq, n)
    pub b_ineq: Vec<f64>,      // Inequality RHS (m_ineq,)
    pub a_eq: Vec<Vec<f64>>,   // Equality constraint matrix (m_eq, n)
    pub b_eq: Vec<f64>,        // Equality RHS (m_eq,)
    pub lb: Vec<f64>,          // Lower bounds (n,)
    pub ub: Vec<f64>,          // Upper bounds (n,)
}

impl LpProblem {
    /// Create a new LP problem with validation
    pub fn new(
        c: Vec<f64>,
        a_ineq: Vec<Vec<f64>>,
        b_ineq: Vec<f64>,
        a_eq: Vec<Vec<f64>>,
        b_eq: Vec<f64>,
        lb: Vec<f64>,
        ub: Vec<f64>,
    ) -> Result<Self> {
        let n = c.len();

        // Validate dimensions
        if lb.len() != n || ub.len() != n {
            return Err(BatchLpError::InvalidDimensions(format!(
                "Bounds dimension mismatch: c={}, lb={}, ub={}",
                n,
                lb.len(),
                ub.len()
            )));
        }

        if !a_ineq.is_empty() {
            if a_ineq.len() != b_ineq.len() {
                return Err(BatchLpError::InvalidDimensions(format!(
                    "Inequality constraint mismatch: A rows={}, b={}",
                    a_ineq.len(),
                    b_ineq.len()
                )));
            }
            for (i, row) in a_ineq.iter().enumerate() {
                if row.len() != n {
                    return Err(BatchLpError::InvalidDimensions(format!(
                        "Inequality constraint row {} has {} columns, expected {}",
                        i,
                        row.len(),
                        n
                    )));
                }
            }
        }

        if !a_eq.is_empty() {
            if a_eq.len() != b_eq.len() {
                return Err(BatchLpError::InvalidDimensions(format!(
                    "Equality constraint mismatch: A_eq rows={}, b_eq={}",
                    a_eq.len(),
                    b_eq.len()
                )));
            }
            for (i, row) in a_eq.iter().enumerate() {
                if row.len() != n {
                    return Err(BatchLpError::InvalidDimensions(format!(
                        "Equality constraint row {} has {} columns, expected {}",
                        i,
                        row.len(),
                        n
                    )));
                }
            }
        }

        Ok(LpProblem {
            c,
            a_ineq,
            b_ineq,
            a_eq,
            b_eq,
            lb,
            ub,
        })
    }

    /// Solve this LP problem
    pub fn solve(&self) -> Result<LpSolution> {
        let n = self.c.len();
        let m_ineq = self.b_ineq.len();
        let m_eq = self.b_eq.len();

        // Create a row-wise problem
        let mut problem = RowProblem::new();

        // Add variables with bounds and objective coefficients
        let cols: Vec<Col> = (0..n)
            .map(|i| problem.add_column(self.c[i], self.lb[i]..self.ub[i]))
            .collect();

        // Add inequality constraints: A x <= b
        for i in 0..m_ineq {
            let row_factors: Vec<(Col, f64)> = self.a_ineq[i]
                .iter()
                .enumerate()
                .filter(|(_, &coeff)| coeff != 0.0)
                .map(|(j, &coeff)| (cols[j], coeff))
                .collect();

            // Add constraint: sum <= b_ineq[i]
            problem.add_row(..=self.b_ineq[i], &row_factors);
        }

        // Add equality constraints: A_eq x == b_eq
        for i in 0..m_eq {
            let row_factors: Vec<(Col, f64)> = self.a_eq[i]
                .iter()
                .enumerate()
                .filter(|(_, &coeff)| coeff != 0.0)
                .map(|(j, &coeff)| (cols[j], coeff))
                .collect();

            // Add constraint: sum == b_eq[i]
            problem.add_row(self.b_eq[i]..=self.b_eq[i], &row_factors);
        }

        // Solve
        let solved = problem.optimise(Sense::Minimise).solve();

        // Extract solution
        let status = match solved.status() {
            highs::HighsModelStatus::Optimal => SolutionStatus::Optimal,
            highs::HighsModelStatus::Infeasible => SolutionStatus::Infeasible,
            highs::HighsModelStatus::Unbounded | highs::HighsModelStatus::UnboundedOrInfeasible => {
                SolutionStatus::Unbounded
            }
            _ => SolutionStatus::Error,
        };

        if status == SolutionStatus::Optimal {
            let solution = solved.get_solution();
            let x = solution.columns().to_vec();
            let objective = solved.objective_value();

            Ok(LpSolution {
                status,
                objective: Some(objective),
                x: Some(x),
            })
        } else {
            Ok(LpSolution {
                status,
                objective: None,
                x: None,
            })
        }
    }
}

/// Solve multiple LP problems in parallel using default thread count
pub fn solve_batch(problems: &[LpProblem]) -> Vec<Result<LpSolution>> {
    problems.par_iter().map(|problem| problem.solve()).collect()
}

/// Solve multiple LP problems in parallel with custom thread count
pub fn solve_batch_with_threads(
    problems: &[LpProblem],
    num_threads: usize,
) -> Vec<Result<LpSolution>> {
    // Build a custom thread pool with specified number of threads
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    // Use the custom pool to solve
    pool.install(|| problems.par_iter().map(|problem| problem.solve()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_lp() {
        // min -x - y
        // s.t. x + y <= 2
        //      x, y >= 0
        let problem = LpProblem::new(
            vec![-1.0, -1.0],                   // c
            vec![vec![1.0, 1.0]],               // A
            vec![2.0],                          // b
            vec![],                             // A_eq
            vec![],                             // b_eq
            vec![0.0, 0.0],                     // lb
            vec![f64::INFINITY, f64::INFINITY], // ub
        )
        .unwrap();

        let solution = problem.solve().unwrap();
        assert_eq!(solution.status, SolutionStatus::Optimal);
        assert!((solution.objective.unwrap() - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_batch_solve() {
        let problem1 = LpProblem::new(
            vec![1.0, 1.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![1.0, 1.0],
            vec![],
            vec![],
            vec![0.0, 0.0],
            vec![f64::INFINITY, f64::INFINITY],
        )
        .unwrap();

        let problem2 = LpProblem::new(
            vec![-1.0, -1.0],
            vec![vec![1.0, 1.0]],
            vec![2.0],
            vec![],
            vec![],
            vec![0.0, 0.0],
            vec![f64::INFINITY, f64::INFINITY],
        )
        .unwrap();

        let solutions = solve_batch(&[problem1, problem2]);
        assert_eq!(solutions.len(), 2);
        assert!(solutions[0].is_ok());
        assert!(solutions[1].is_ok());
    }
}
