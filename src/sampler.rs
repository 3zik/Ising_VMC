use crate::SpinConfig;
use crate::rbm::Rbm;
use nalgebra::DVector;
use rand::Rng;

// ---------------------------------------------------------------------------
// MetropolisSampler
// ---------------------------------------------------------------------------
// Samples spin configurations from P(s) = |psi(s)|^2 using single spin-flip
// Metropolis-Hastings proposals.
//
// Algorithm per step:
//   1. Pick a random site i
//   2. Compute log_ratio = log|psi(s_flipped)| - log|psi(s)|
//   3. Accept with probability min(1, exp(2 * log_ratio))
//      (factor 2: we sample from |psi|^2, not |psi|)
//   4. If accepted: flip spin and update theta cache incrementally
//      If rejected: keep current state unchanged
//
// theta_j = b_j + sum_i W_ij * s_i is maintained incrementally.
// When spin i flips: theta_j -= 2 * s_i * W_ij  for all j.
// This keeps each update O(M) rather than O(N*M).

pub struct MetropolisSampler {
    pub config: SpinConfig,
    pub theta: DVector<f64>,
    pub n_accepted: usize,
    pub n_proposed: usize,
}

impl MetropolisSampler {
    /// Initialise with a random spin configuration.
    pub fn new(n: usize, rbm: &Rbm, rng: &mut impl Rng) -> Self {
        let config = SpinConfig::random_init(n, rng);
        let theta = rbm.theta(&config.spins);
        MetropolisSampler { config, theta, n_accepted: 0, n_proposed: 0 }
    }

    /// Attempt one single spin-flip Metropolis step.
    pub fn step(&mut self, rbm: &Rbm, rng: &mut impl Rng) -> bool {
        let site = rng.gen_range(0..self.config.len());
        let log_ratio = rbm.log_ratio(&self.config.spins, site, &self.theta);
        let log_accept = 2.0 * log_ratio;

        self.n_proposed += 1;

        let accepted = if log_accept >= 0.0 {
            true
        } else {
            rng.gen::<f64>() < log_accept.exp()
        };

        if accepted {
            // Update theta incrementally before flipping
            let si = self.config.spins[site] as f64;
            for j in 0..rbm.n_hidden {
                self.theta[j] -= 2.0 * si * rbm.w[(site, j)];
            }
            self.config.flip(site);
            self.n_accepted += 1;
        }

        accepted
    }

    /// Run `n_steps` burn-in steps, then reset acceptance counters.
    pub fn thermalize(&mut self, n_steps: usize, rbm: &Rbm, rng: &mut impl Rng) {
        for _ in 0..n_steps {
            self.step(rbm, rng);
        }
        self.n_accepted = 0;
        self.n_proposed = 0;
    }

    /// Collect `n_samples` configurations, one step between each sample.
    pub fn collect_samples(
        &mut self,
        n_samples: usize,
        rbm: &Rbm,
        rng: &mut impl Rng,
    ) -> Vec<(Vec<i8>, DVector<f64>)> {
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            self.step(rbm, rng);
            samples.push((self.config.spins.clone(), self.theta.clone()));
        }
        samples
    }

    /// Fraction of proposals accepted since last reset.
    pub fn acceptance_rate(&self) -> f64 {
        if self.n_proposed == 0 { return 0.0; }
        self.n_accepted as f64 / self.n_proposed as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rbm::Rbm;
    use nalgebra::{DMatrix, DVector};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_sampler(n: usize, seed: u64) -> (MetropolisSampler, Rbm, StdRng) {
        let mut rng = StdRng::seed_from_u64(seed);
        let rbm = Rbm::new_random(n, n, 0.1, &mut rng);
        let sampler = MetropolisSampler::new(n, &rbm, &mut rng);
        (sampler, rbm, rng)
    }

    // theta cache must stay in sync with the spin config after every step.
    #[test]
    fn test_theta_cache_stays_in_sync() {
        let (mut sampler, rbm, mut rng) = make_sampler(8, 42);
        for _ in 0..200 {
            sampler.step(&rbm, &mut rng);
            let expected = rbm.theta(&sampler.config.spins);
            let diff = (&sampler.theta - &expected).amax();
            assert!(diff < 1e-10, "theta cache out of sync: max diff = {diff:.2e}");
        }
    }

    // Acceptance rate should be in a reasonable range.
    #[test]
    fn test_acceptance_rate_reasonable() {
        let (mut sampler, rbm, mut rng) = make_sampler(8, 7);
        sampler.thermalize(500, &rbm, &mut rng);
        for _ in 0..1000 { sampler.step(&rbm, &mut rng); }
        let rate = sampler.acceptance_rate();
        assert!(rate > 0.1 && rate < 1.0,
            "acceptance rate {rate:.3} out of expected range (0.1, 1.0)");
    }

    // collect_samples must return exactly n_samples pairs.
    #[test]
    fn test_collect_samples_count() {
        let (mut sampler, rbm, mut rng) = make_sampler(8, 13);
        let samples = sampler.collect_samples(50, &rbm, &mut rng);
        assert_eq!(samples.len(), 50);
    }

    // Each collected sample's theta must match a fresh recompute.
    #[test]
    fn test_collected_samples_theta_correct() {
        let (mut sampler, rbm, mut rng) = make_sampler(8, 99);
        sampler.thermalize(200, &rbm, &mut rng);
        let samples = sampler.collect_samples(20, &rbm, &mut rng);
        for (spins, theta) in &samples {
            let expected = rbm.theta(spins);
            let diff = (theta - &expected).amax();
            assert!(diff < 1e-10, "sample theta wrong: max diff = {diff:.2e}");
        }
    }

    // thermalize() must reset the acceptance counters.
    #[test]
    fn test_thermalize_resets_counters() {
        let (mut sampler, rbm, mut rng) = make_sampler(8, 55);
        for _ in 0..100 { sampler.step(&rbm, &mut rng); }
        sampler.thermalize(100, &rbm, &mut rng);
        assert_eq!(sampler.n_proposed, 0);
        assert_eq!(sampler.n_accepted, 0);
    }

    // All spins in every collected sample must be +-1.
    #[test]
    fn test_samples_are_valid_configs() {
        let (mut sampler, rbm, mut rng) = make_sampler(8, 77);
        let samples = sampler.collect_samples(100, &rbm, &mut rng);
        for (spins, _) in &samples {
            assert!(spins.iter().all(|&s| s == 1 || s == -1));
        }
    }

    // EXIT CRITERION: at h/J >> 1 the distribution is nearly uniform, so
    // the mean of sz = (1/N) sum_i s_i should be near zero.
    // We test this with a zero-weight RBM (exactly uniform P(s)).
    #[test]
    fn test_mean_sz_near_zero_uniform_distribution() {
        let n = 8;
        let m = 8;
        let rbm = Rbm {
            a: DVector::zeros(n),
            b: DVector::zeros(m),
            w: DMatrix::zeros(n, m),
            n_visible: n,
            n_hidden: m,
        };

        let mut rng = StdRng::seed_from_u64(123);
        let mut sampler = MetropolisSampler::new(n, &rbm, &mut rng);
        sampler.thermalize(500, &rbm, &mut rng);

        let n_samples = 5000;
        let samples = sampler.collect_samples(n_samples, &rbm, &mut rng);

        let mean_sz: f64 = samples.iter()
            .map(|(spins, _)| spins.iter().map(|&s| s as f64).sum::<f64>() / n as f64)
            .sum::<f64>() / n_samples as f64;

        assert!(
            mean_sz.abs() < 0.1,
            "mean sz = {mean_sz:.4}, expected ~0 for uniform distribution"
        );
    }
}
