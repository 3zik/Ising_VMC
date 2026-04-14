use crate::SpinConfig;
use crate::rbm::Rbm;
use nalgebra::DVector;
use rand::Rng;

pub struct MetropolisSampler {
    pub config: SpinConfig,
    pub theta: DVector<f64>,
    pub n_accepted: usize,
    pub n_proposed: usize,
}
impl MetropolisSampler {
    pub fn new(n: usize, rbm: &Rbm, rng: &mut impl Rng) -> Self {
        let config = SpinConfig::random_init(n, rng);
        let theta = rbm.theta(&config.spins);
        MetropolisSampler { config, theta, n_accepted: 0, n_proposed: 0 }
    }
    pub fn step(&mut self, rbm: &Rbm, rng: &mut impl Rng) -> bool {
        let site = rng.gen_range(0..self.config.len());
        let log_ratio = rbm.log_ratio(&self.config.spins, site, &self.theta);
        self.n_proposed += 1;
        let accepted = log_ratio*2. >= 0. || rng.gen::<f64>() < (log_ratio*2.).exp();
        if accepted {
            let si = self.config.spins[site] as f64;
            for j in 0..rbm.n_hidden { self.theta[j] -= 2.*si*rbm.w[(site,j)]; }
            self.config.flip(site);
            self.n_accepted += 1;
        }
        accepted
    }
    pub fn thermalize(&mut self, n_steps: usize, rbm: &Rbm, rng: &mut impl Rng) {
        for _ in 0..n_steps { self.step(rbm, rng); }
        self.n_accepted = 0; self.n_proposed = 0;
    }
    pub fn collect_samples(&mut self, n: usize, rbm: &Rbm, rng: &mut impl Rng) -> Vec<(Vec<i8>, DVector<f64>)> {
        (0..n).map(|_| { self.step(rbm,rng); (self.config.spins.clone(), self.theta.clone()) }).collect()
    }
    pub fn acceptance_rate(&self) -> f64 {
        if self.n_proposed==0 { return 0.; }
        self.n_accepted as f64 / self.n_proposed as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rbm::Rbm;
    use nalgebra::{DMatrix, DVector};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make(n: usize, seed: u64) -> (MetropolisSampler, Rbm, StdRng) {
        let mut rng = StdRng::seed_from_u64(seed);
        let rbm = Rbm::new_random(n,n,0.1,&mut rng);
        let s = MetropolisSampler::new(n,&rbm,&mut rng);
        (s, rbm, StdRng::seed_from_u64(seed+1000))
    }
    #[test] fn test_theta_cache_sync() {
        let (mut s, rbm, mut rng) = make(8,42);
        for _ in 0..200 {
            s.step(&rbm,&mut rng);
            assert!((&s.theta - rbm.theta(&s.config.spins)).amax() < 1e-10);
        }
    }
    #[test] fn test_acceptance_rate() {
        let (mut s, rbm, mut rng) = make(8,7);
        s.thermalize(500,&rbm,&mut rng);
        for _ in 0..1000 { s.step(&rbm,&mut rng); }
        let r = s.acceptance_rate();
        assert!(r>0.1&&r<1.,"rate {r:.3}");
    }
    #[test] fn test_mean_sz_uniform() {
        let n=8; let m=8;
        let rbm = Rbm { a:DVector::zeros(n), b:DVector::zeros(m),
            w:DMatrix::zeros(n,m), n_visible:n, n_hidden:m };
        let mut rng = StdRng::seed_from_u64(123);
        let mut s = MetropolisSampler::new(n,&rbm,&mut rng);
        s.thermalize(500,&rbm,&mut rng);
        let samps = s.collect_samples(5000,&rbm,&mut rng);
        let mean: f64 = samps.iter()
            .map(|(sp,_)| sp.iter().map(|&x| x as f64).sum::<f64>()/n as f64)
            .sum::<f64>() / 5000.;
        assert!(mean.abs()<0.1,"mean sz={mean:.4}");
    }
}
