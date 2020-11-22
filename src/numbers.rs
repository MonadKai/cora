//! # Real Number
//! Most algorithms in cora rely on basic linear operations like dot product.
//! This module defines real number and some useful functions that are used in [Linear Algebra](../linalg/index.html) module.

use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use num_traits::{Float, FromPrimitive};
use rand::Rng;

/// Defines real number
/// <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
pub trait Real:
    Float
    + FromPrimitive
    + Debug
    + Display
    + Copy
    + Sum
    + Product
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Copy sign from `sign` - another real number
    fn copysign(self, sign: Self) -> Self;

    /// Caculate natural \\( \ln(1 + e^x) \\) without overflow.
    fn ln_1pe(self) -> Self;

    /// Efficient implementation of sigmoid function, \\( S(x) = \frac{1}{1 + e^{-x}} \\), see [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)
    fn sigmoid(self) -> Self;

    /// Return psudorandom number between 0 and 1
    fn rand() -> Self;

    /// Return 2
    fn two() -> Self;

    /// Return .5
    fn half() -> Self;

    /// Return \\( x^2 \\)
    fn square(self) -> Self {
        self * self
    }

    /// Raw transmutation to u64
    fn to_f32_bits(self) -> u32;
}

impl Real for f64 {
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }

    fn ln_1pe(self) -> Self {
        // avoid overflow
        if self > 15. {
            self
        } else {
            self.exp().ln_1p()
        }
    }

    fn sigmoid(self) -> Self {
        // error less than eps
        if self < -40. {
            0.
        } else if self > 40. {
            1.
        } else {
            1. / (1. + f64::exp(-self))
        }
    }

    fn rand() -> Self {
        let mut rng = rand::thread_rng();
        rng.gen()
    }

    fn two() -> Self {
        2f64
    }

    fn half() -> Self {
        0.5f64
    }

    fn to_f32_bits(self) -> u32 {
        self.to_bits() as u32
    }
}

impl Real for f32 {
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }

    fn ln_1pe(self) -> Self {
        // avoid overflow
        if self > 15. {
            self
        } else {
            self.exp().ln_1p()
        }
    }

    fn sigmoid(self) -> Self {
        // error less than eps
        if self < -40. {
            0.
        } else if self > 40. {
            1.
        } else {
            1. / (1. + f32::exp(-self))
        }
    }

    fn rand() -> Self {
        let mut rng = rand::thread_rng();
        rng.gen()
    }

    fn two() -> Self {
        2f32
    }

    fn half() -> Self {
        0.5f32
    }

    fn to_f32_bits(self) -> u32 {
        self.to_bits() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid() {
        assert_eq!(1.0.sigmoid(), 0.7310585786300049);
        assert_eq!(41.0.sigmoid(), 1.);
        assert_eq!((-41.0).sigmoid(), 0.);
    }
}
