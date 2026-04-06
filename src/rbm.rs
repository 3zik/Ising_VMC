use nalgebra::{DMatrix, DVector};
use rand::Rng;

// ---------------------------------------------------------------------------
// RBM
// ---------------------------------------------------------------------------
// Represents the wavefunction as a Restricted Boltzmann Machine:
//
//   psi(s) = exp(sum_i a_i * s_i) * prod_j 2*cosh(theta_j(s))
//
// where theta_j(s) = b_j + sum_i W_ij * s_i
//
// We always work in log-space to avoid overflow:
//
//   log|psi(s)| = sum_i a_i*s_i + sum_j log(2*cosh(theta_j(s)))
//
// Parameters:
//   a  : visible biases,  shape (n_visible,)
//   b  : hidden biases,   shape (n_hidden,)
//   W  : weight matrix,   shape (n_visible, n_hidden)  W[(i,j)] = W_ij

pub struct Rbm {
    pub a: DVector<f64>, // visible biases
    pub b: DVector<f64>, // hidden biases
    pub w: DMatrix<f64>, // weight matrix, w[(i,j)]
    pub n_visible: usize,
    pub n_hidden: usize,
}

impl Rbm {
    /// Create an RBM with small random parameters.
    /// std controls the scale of the initial weights (typically 0.01 - 0.1).
    pub fn new_random(n_visible: usize, n_hidden: usize, std: f64, rng: &mut impl Rng) -> Self {
        let a = DVector::from_fn(n_visible, |_, _| rng.gen::<f64>() * std * 2.0 - std);
        let b = DVector::from_fn(n_hidden,  |_, _| rng.gen::<f64>() * std * 2.0 - std);
        let w = DMatrix::from_fn(n_visible, n_hidden, |_, _| rng.gen::<f64>() * std * 2.0 - std);
        Rbm { a, b, w, n_visible, n_hidden }
    }

    /// Compute the hidden-unit pre-activations theta_j(s) for all j.
    /// theta_j(s) = b_j + sum_i W_ij * s_i
    /// Shape: (n_hidden,)
    pub fn theta(&self, s: &[i8]) -> DVector<f64> {
        let s_f: DVector<f64> = DVector::from_fn(self.n_visible, |i, _| s[i] as f64);
        &self.b + self.w.transpose() * s_f
    }

    /// log|psi(s)| = sum_i a_i*s_i + sum_j log(2*cosh(theta_j(s)))
    pub fn log_amplitude(&self, s: &[i8]) -> f64 {
        let visible_term: f64 = self.a.iter().zip(s.iter()).map(|(a, &si)| a * si as f64).sum();
        let theta = self.theta(s);
        let hidden_term: f64 = theta.iter().map(|&t| log2cosh(t)).sum();
        visible_term + hidden_term
    }

    /// log(|psi(s_flipped)| / |psi(s)|) when only spin at `site` is flipped.
    ///
    /// Flipping spin i changes:
    ///   visible term: a_i * s_i  ->  a_i * (-s_i)   delta = -2 * a_i * s_i
    ///   theta_j:      theta_j    ->  theta_j - 2*s_i*W_ij  for all j
    ///
    /// So log_ratio = -2*a_site*s_site
    ///              + sum_j [ log2cosh(theta_j - 2*s_site*W_site_j)
    ///                       - log2cosh(theta_j) ]
    ///
    /// This is O(M) rather than O(N*M) -- only theta values update.
    pub fn log_ratio(&self, s: &[i8], site: usize, theta: &DVector<f64>) -> f64 {
        let si = s[site] as f64;
        let visible_delta = -2.0 * self.a[site] * si;
        let hidden_delta: f64 = (0..self.n_hidden)
            .map(|j| {
                let theta_new = theta[j] - 2.0 * si * self.w[(site, j)];
                log2cosh(theta_new) - log2cosh(theta[j])
            })
            .sum();
        visible_delta + hidden_delta
    }

    /// Log-derivatives of log|psi(s)| with respect to all parameters.
    /// These are the O vectors used in the VMC gradient estimator.
    ///
    ///   d log|psi| / d a_i  = s_i
    ///   d log|psi| / d b_j  = tanh(theta_j(s))
    ///   d log|psi| / d W_ij = s_i * tanh(theta_j(s))
    ///
    /// Returns (grad_a, grad_b, grad_W) as flat structures.
    pub fn log_derivatives(&self, s: &[i8], theta: &DVector<f64>) -> LogDerivatives {
        let grad_a = DVector::from_fn(self.n_visible, |i, _| s[i] as f64);
        let tanh_theta = DVector::from_fn(self.n_hidden, |j, _| theta[j].tanh());
        let grad_b = tanh_theta.clone();
        let grad_w = DMatrix::from_fn(self.n_visible, self.n_hidden, |i, j| {
            s[i] as f64 * tanh_theta[j]
        });
        LogDerivatives { grad_a, grad_b, grad_w }
    }

    /// Total number of parameters (for gradient vector sizing).
    pub fn n_params(&self) -> usize {
        self.n_visible + self.n_hidden + self.n_visible * self.n_hidden
    }
}

// ---------------------------------------------------------------------------
// LogDerivatives
// ---------------------------------------------------------------------------
// Holds d log|psi| / d theta for all parameter groups.

pub struct LogDerivatives {
    pub grad_a: DVector<f64>,
    pub grad_b: DVector<f64>,
    pub grad_w: DMatrix<f64>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Numerically stable log(2 * cosh(x)).
/// For large |x|, cosh(x) overflows, so we use:
///   log(2*cosh(x)) = |x| + log(1 + exp(-2|x|))
///                  ≈ |x|  for large |x|
#[inline]
pub fn log2cosh(x: f64) -> f64 {
    let ax = x.abs();
    ax + (1.0 + (-2.0 * ax).exp()).ln()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn test_rbm() -> (Rbm, StdRng) {
        let mut rng = StdRng::seed_from_u64(42);
        let rbm = Rbm::new_random(4, 4, 0.1, &mut rng);
        (rbm, rng)
    }

    fn test_config() -> Vec<i8> {
        vec![1, -1, 1, 1]
    }

    // log_ratio must agree with recomputing log_amplitude from scratch
    // after the flip. This is the key correctness test.
    #[test]
    fn test_log_ratio_matches_recompute() {
        let (rbm, _) = test_rbm();
        let s = test_config();
        let theta = rbm.theta(&s);

        for site in 0..s.len() {
            let fast_ratio = rbm.log_ratio(&s, site, &theta);

            // Naive: flip, recompute, compare
            let mut s_flipped = s.clone();
            s_flipped[site] *= -1;
            let naive_ratio = rbm.log_amplitude(&s_flipped) - rbm.log_amplitude(&s);

            assert!(
                (fast_ratio - naive_ratio).abs() < 1e-10,
                "site {site}: fast={fast_ratio:.12}, naive={naive_ratio:.12}"
            );
        }
    }

    // log_ratio must work for every spin in a random config
    #[test]
    fn test_log_ratio_random_config() {
        let (rbm, mut rng) = test_rbm();
        let s: Vec<i8> = (0..4).map(|_| if rng.gen_bool(0.5) { 1 } else { -1 }).collect();
        let theta = rbm.theta(&s);

        for site in 0..s.len() {
            let fast = rbm.log_ratio(&s, site, &theta);
            let mut sf = s.clone();
            sf[site] *= -1;
            let naive = rbm.log_amplitude(&sf) - rbm.log_amplitude(&s);
            assert!((fast - naive).abs() < 1e-10, "site {site}: {fast:.12} vs {naive:.12}");
        }
    }

    // log2cosh must be numerically stable for large inputs
    #[test]
    fn test_log2cosh_large_values() {
        // For x=100, naive 2*cosh(100) overflows to inf; our version must not.
        let result = log2cosh(100.0);
        assert!(result.is_finite(), "log2cosh(100) should be finite, got {result}");
        assert!((result - 100.0).abs() < 1e-10, "log2cosh(100) should be ~100");

        let result_neg = log2cosh(-100.0);
        assert!((result_neg - result).abs() < 1e-14, "log2cosh should be even");
    }

    // log_derivatives: grad_a_i should equal s_i
    #[test]
    fn test_log_derivatives_grad_a() {
        let (rbm, _) = test_rbm();
        let s = test_config();
        let theta = rbm.theta(&s);
        let derivs = rbm.log_derivatives(&s, &theta);
        for i in 0..s.len() {
            assert_eq!(derivs.grad_a[i], s[i] as f64, "grad_a[{i}] should equal s[{i}]");
        }
    }

    // log_derivatives: check grad_W numerically
    #[test]
    fn test_log_derivatives_grad_w_numerical() {
        let (rbm, _) = test_rbm();
        let s = test_config();
        let theta = rbm.theta(&s);
        let derivs = rbm.log_derivatives(&s, &theta);
        let eps = 1e-5;

        // Check a few entries of grad_W numerically
        for (i, j) in [(0, 0), (1, 2), (3, 1)] {
            let mut rbm_p = Rbm::new_random(4, 4, 0.1, &mut StdRng::seed_from_u64(42));
            let mut rbm_m = Rbm::new_random(4, 4, 0.1, &mut StdRng::seed_from_u64(42));
            rbm_p.w[(i, j)] += eps;
            rbm_m.w[(i, j)] -= eps;
            let numerical = (rbm_p.log_amplitude(&s) - rbm_m.log_amplitude(&s)) / (2.0 * eps);
            let analytical = derivs.grad_w[(i, j)];
            assert!(
                (numerical - analytical).abs() < 1e-6,
                "grad_W[{i},{j}]: numerical={numerical:.8}, analytical={analytical:.8}"
            );
        }
    }

    // n_params should equal n_visible + n_hidden + n_visible*n_hidden
    #[test]
    fn test_n_params() {
        let (rbm, _) = test_rbm();
        assert_eq!(rbm.n_params(), 4 + 4 + 4 * 4);
    }
}
