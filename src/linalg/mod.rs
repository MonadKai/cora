use crate::numbers::Real;
use std::fmt::Debug;

/// Column or row vector
pub trait BaseVector<T: Real>: Clone + Debug {
    /// Get an element of a vector
    /// * `i` - index of an element
    fn get(&self, i: usize) -> T;

    /// Set an element at `i` to `x`
    /// * `i` - index of an element
    /// * `x` - new value
    fn set(&mut self, i: usize, x: T);

    /// Get number of element in the vector
    fn len(&self) -> usize;

    /// Return true if the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a new vector from a &[T]
    fn from_array(arr: &[T]) -> Self {
        let mut v = Self::zeros(arr.len());
        for (i, elem) in arr.iter().enumerate() {
            v.set(i, *elem);
        }
        v
    }

    /// Return a vector with the elements of the one-dimensional array.
    fn to_vec(&self) -> Vec<T>;

    /// Create new vector with zeros of size `len`.
    fn zeros(len: usize) -> Self;

    /// Create new vector with ones of size `len`.
    fn ones(len: usize) -> Self;

    /// Create new vector with `len` where each element is set to `value`.
    fn fill(len: usize, value: T) -> Self;

    /// Vector dot product
    fn dot(&self, other: &Self) -> T;

    /// Return true if matrices are element-wise equal within a tolerance `eps`/
    fn approximate_eq(&self, other: &Self, eps: T) -> bool;

    /// Return [L2 norm](https://en.wikipedia.org/wiki/Matrix_norm) of the vector.
    fn norm2(&self) -> T;

    /// Return [vectors norm](https://en.wikipedia.org/wiki/Matrix_norm) of order `p`.
    fn norm(&self, p: T) -> T;

    /// Divide single element of the vector by `x`, write result to original vector.
    fn div_element_mut(&mut self, pos: usize, x: T);

    /// Multiply single element of the vector by `x`, write result to original vector.
    fn mul_element_mut(&mut self, pos: usize, x: T);

    /// Add single element of the vector to `x`, write result to original vector.
    fn add_element_mut(&mut self, pos: usize, x: T);

    /// Subtract `x` from single element of the vector, write result to the original vector.
    fn sub_element_mut(&mut self, pos: usize, x: T);

    /// Add vectors, element-wise, overriding original vector with result.
    fn add_mut(&mut self, other: &Self) -> &Self;

    /// Subtract vectors, element-wise, overriding original vector with result.
    fn sub_mut(&mut self, other: &Self) -> &Self;

    /// Multiply vectors, element-wise, overriding original vector with result.
    fn mul_mut(&mut self, other: &Self) -> &Self;

    /// Divide vectors, element-wise, overriding original vector with result.
    fn div_mut(&mut self, other: &Self) -> &Self;

    /// Add vectors, element-wise
    fn add(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.add_mut(other);
        r
    }

    /// Subtract vectors, element-wise
    fn sub(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.sub_mut(other);
        r
    }

    /// Multiply vectors, element-wise
    fn mul(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.mul_mut(other);
        r
    }

    /// Divide vectors, element-wise
    fn div(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.div_mut(other);
        r
    }

    /// Calculates sum of all elements of the vector.
    fn sum(&self) -> T;

    /// Returns unique values from the vector.
    fn unique(&self) -> Vec<T>;

    /// Compute the arithmetic mean.
    fn mean(&self) -> T {
        self.sum() / T::from_usize(self.len()).unwrap()
    }

    /// Compute the variance.
    fn var(&self) -> T {
        let n = self.len();
        let mut mu = T::zero();
        let mut sum = T::zero();
        let div = T::from_usize(n).unwrap();
        for i in 0..n {
            let xi = self.get(i);
            mu += xi;
            sum += xi * xi;
        }
        mu /= div;
        sum / div - mu * mu
    }

    /// Compute the standard deviation.
    fn std(&self) -> T {
        self.var().sqrt()
    }
}
