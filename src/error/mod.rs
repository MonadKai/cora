use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Failure {
    err: FailedError,
    msg: String,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum FailedError {
    /// Can not fit algorithm to data
    FitFailed = 1,
    /// Can not predict new values
    PredictFailed,
    /// Can not transform data
    TransformFailed,
    /// Can not find an item
    FindFailed,
    /// Can not decompose a matrix
    DecompositionFailed,
    /// Can not solve for X
    SolutionFailed,
}

impl Failure {
    /// get type of error
    #[inline]
    pub fn error(&self) -> FailedError {
        self.err
    }

    /// new instance of `FailedError::FitError'
    pub fn fit(msg: &str) -> Self {
        Failure {
            err: FailedError::FitFailed,
            msg: msg.to_owned(),
        }
    }

    /// new instance of `FailedError::PredictFailed`
    pub fn predict(msg: &str) -> Self {
        Failure {
            err: FailedError::PredictFailed,
            msg: msg.to_owned(),
        }
    }

    /// new instance of `FailedError::TransformFailed`
    pub fn transform(msg: &str) -> Self {
        Failure {
            err: FailedError::TransformFailed,
            msg: msg.to_owned(),
        }
    }

    /// new instance of `err`
    pub fn because(err: FailedError, msg: &str) -> Self {
        Failure {
            err,
            msg: msg.to_owned(),
        }
    }
}

impl PartialEq for FailedError {
    #[inline(always)]
    fn eq(&self, rhs: &Self) -> bool {
        *self as u8 == *rhs as u8
    }
}

impl PartialEq for Failure {
    fn eq(&self, rhs: &Self) -> bool {
        self.err == rhs.err && self.msg == rhs.msg
    }
}

impl fmt::Display for FailedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let failed_err_str = match self {
            FailedError::FitFailed => "Fit failed",
            FailedError::PredictFailed => "Predict failed",
            FailedError::TransformFailed => "Transform failed",
            FailedError::FindFailed => "Find failed",
            FailedError::DecompositionFailed => "Decomposition failed",
            FailedError::SolutionFailed => "Can not find solution",
        };
        write!(f, "{}", failed_err_str)
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.err, self.msg)
    }
}

impl Error for Failure {}
