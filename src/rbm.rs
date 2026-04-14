use nalgebra::{DMatrix, DVector};
use rand::Rng;

pub struct Rbm {
    pub a: DVector<f64>,
    pub b: DVector<f64>,
    pub w: DMatrix<f64>,
    pub n_visible: usize,
    pub n_hidden: usize,
}

impl Rbm {
    pub fn new_random(n_visible: usize, n_hidden: usize, std: f64, rng: &mut impl Rng) -> Self {
        let a = DVector::from_fn(n_visible, |_,_| rng.gen::<f64>()*std*2.-std);
        let b = DVector::from_fn(n_hidden,  |_,_| rng.gen::<f64>()*std*2.-std);
        let w = DMatrix::from_fn(n_visible, n_hidden, |_,_| rng.gen::<f64>()*std*2.-std);
        Rbm { a, b, w, n_visible, n_hidden }
    }
    pub fn theta(&self, s: &[i8]) -> DVector<f64> {
        let sf = DVector::from_fn(self.n_visible, |i,_| s[i] as f64);
        &self.b + self.w.transpose() * sf
    }
    pub fn log_amplitude(&self, s: &[i8]) -> f64 {
        let vis: f64 = self.a.iter().zip(s).map(|(a,&si)| a*si as f64).sum();
        let hid: f64 = self.theta(s).iter().map(|&t| log2cosh(t)).sum();
        vis + hid
    }
    pub fn log_ratio(&self, s: &[i8], site: usize, theta: &DVector<f64>) -> f64 {
        let si = s[site] as f64;
        let vis = -2.*self.a[site]*si;
        let hid: f64 = (0..self.n_hidden).map(|j| {
            log2cosh(theta[j] - 2.*si*self.w[(site,j)]) - log2cosh(theta[j])
        }).sum();
        vis + hid
    }
    pub fn log_derivatives(&self, s: &[i8], theta: &DVector<f64>) -> LogDerivatives {
        let grad_a = DVector::from_fn(self.n_visible, |i,_| s[i] as f64);
        let tanh_t = DVector::from_fn(self.n_hidden, |j,_| theta[j].tanh());
        let grad_b = tanh_t.clone();
        let grad_w = DMatrix::from_fn(self.n_visible, self.n_hidden, |i,j| s[i] as f64*tanh_t[j]);
        LogDerivatives { grad_a, grad_b, grad_w }
    }
    pub fn n_params(&self) -> usize { self.n_visible + self.n_hidden + self.n_visible*self.n_hidden }
}

pub struct LogDerivatives {
    pub grad_a: DVector<f64>,
    pub grad_b: DVector<f64>,
    pub grad_w: DMatrix<f64>,
}

#[inline]
pub fn log2cosh(x: f64) -> f64 {
    let ax = x.abs();
    ax + (1. + (-2.*ax).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    fn rbm42() -> (Rbm, StdRng) {
        let mut rng = StdRng::seed_from_u64(42);
        (Rbm::new_random(4,4,0.1,&mut rng), StdRng::seed_from_u64(42))
    }
    #[test] fn test_log_ratio_matches_recompute() {
        let (rbm,_) = rbm42();
        let s = vec![1i8,-1,1,1];
        let theta = rbm.theta(&s);
        for site in 0..4 {
            let fast = rbm.log_ratio(&s,site,&theta);
            let mut sf = s.clone(); sf[site]*=-1;
            let naive = rbm.log_amplitude(&sf)-rbm.log_amplitude(&s);
            assert!((fast-naive).abs()<1e-10,"site {site}");
        }
    }
    #[test] fn test_log2cosh_large() {
        assert!(log2cosh(100.).is_finite());
        assert!((log2cosh(100.)-100.).abs()<1e-10);
    }
    #[test] fn test_grad_a_equals_s() {
        let (rbm,_) = rbm42();
        let s = vec![1i8,-1,1,1];
        let d = rbm.log_derivatives(&s,&rbm.theta(&s));
        for i in 0..4 { assert_eq!(d.grad_a[i], s[i] as f64); }
    }
    #[test] fn test_grad_w_numerical() {
        let s = vec![1i8,-1,1,1];
        let mut rng = StdRng::seed_from_u64(42);
        let rbm = Rbm::new_random(4,4,0.1,&mut rng);
        let d = rbm.log_derivatives(&s,&rbm.theta(&s));
        let eps = 1e-5;
        for (i,j) in [(0usize,0usize),(1,2),(3,1)] {
            let mut rp = Rbm::new_random(4,4,0.1,&mut StdRng::seed_from_u64(42));
            let mut rm = Rbm::new_random(4,4,0.1,&mut StdRng::seed_from_u64(42));
            rp.w[(i,j)]+=eps; rm.w[(i,j)]-=eps;
            let num = (rp.log_amplitude(&s)-rm.log_amplitude(&s))/(2.*eps);
            assert!((num-d.grad_w[(i,j)]).abs()<1e-6,"W[{i},{j}]");
        }
    }
}
