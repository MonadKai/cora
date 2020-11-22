pub trait BaseEstimator<M, P, E> {
    fn fit(self, x: &M, y: &M, fit_params: P) -> Result<Self, E>
    where
        Self: Sized;
}

pub trait Classifier<M, E> {
    fn predict(self, x: &M) -> Result<M, E>;
}

pub trait Regressor<M, E> {
    fn predict(self, x: &M) -> Result<M, E>;
}
