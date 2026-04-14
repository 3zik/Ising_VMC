use crate::rbm::{Rbm, LogDerivatives};
use crate::sampler::MetropolisSampler;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use csv::Writer;
use std::error::Error;

// ---------------------------------------------------------------------------
// Local energy
// ---------------------------------------------------------------------------
// E_loc(s) = -J * sum_i s_i*s_{i+1}        (diagonal Ising term)
//           + -h * sum_i psi(s^i)/psi(s)    (off-diagonal transverse field)
// where s^i = s with spin i flipped, and psi(s^i)/psi(s) = exp(log_ratio).

pub fn local_energy(spins: &[i8], theta: &DVector<f64>, rbm: &Rbm, j: f64, h: f64) -> f64 {
    let n = spins.len();
    let diagonal: f64 = (0..n-1).map(|i| -j * spins[i] as f64 * spins[i+1] as f64).sum();
    let off_diag: f64 = (0..n).map(|i| -h * rbm.log_ratio(spins,i,theta).exp()).sum();
    diagonal + off_diag
}

// ---------------------------------------------------------------------------
// Gradient estimator
// ---------------------------------------------------------------------------
// dE/dtheta = 2 * ( <E_loc * O> - <E_loc> * <O> )
// where O = d log|psi| / d theta  (log-derivatives from rbm.rs)

pub struct VmcGradient {
    pub grad_a: DVector<f64>,
    pub grad_b: DVector<f64>,
    pub grad_w: DMatrix<f64>,
}

pub fn estimate_gradient(samples: &[(Vec<i8>, DVector<f64>)], rbm: &Rbm, j: f64, h: f64) -> (f64, VmcGradient) {
    let ns = samples.len() as f64;
    let mut e_locs = Vec::with_capacity(samples.len());
    let mut oa: Vec<DVector<f64>> = Vec::new();
    let mut ob: Vec<DVector<f64>> = Vec::new();
    let mut ow: Vec<DMatrix<f64>> = Vec::new();

    for (spins, theta) in samples {
        e_locs.push(local_energy(spins, theta, rbm, j, h));
        let d: LogDerivatives = rbm.log_derivatives(spins, theta);
        oa.push(d.grad_a); ob.push(d.grad_b); ow.push(d.grad_w);
    }

    let e_mean = e_locs.iter().sum::<f64>() / ns;
    let oa_mean = oa.iter().fold(DVector::zeros(rbm.n_visible), |a,x| a+x) / ns;
    let ob_mean = ob.iter().fold(DVector::zeros(rbm.n_hidden),  |a,x| a+x) / ns;
    let ow_mean = ow.iter().fold(DMatrix::zeros(rbm.n_visible, rbm.n_hidden), |a,x| a+x) / ns;

    let mut ga = DVector::zeros(rbm.n_visible);
    let mut gb = DVector::zeros(rbm.n_hidden);
    let mut gw = DMatrix::zeros(rbm.n_visible, rbm.n_hidden);
    for i in 0..samples.len() {
        ga += e_locs[i] * &oa[i];
        gb += e_locs[i] * &ob[i];
        gw += e_locs[i] * &ow[i];
    }
    let grad_a = 2. * (ga/ns - e_mean * &oa_mean);
    let grad_b = 2. * (gb/ns - e_mean * &ob_mean);
    let grad_w = 2. * (gw/ns - e_mean * &ow_mean);

    (e_mean, VmcGradient { grad_a, grad_b, grad_w })
}

// ---------------------------------------------------------------------------
// Adam optimizer
// ---------------------------------------------------------------------------
// m <- beta1*m + (1-beta1)*g
// v <- beta2*v + (1-beta2)*g^2
// theta -= alpha * (m/(1-beta1^t)) / (sqrt(v/(1-beta2^t)) + eps)

pub struct Adam {
    pub alpha: f64, pub beta1: f64, pub beta2: f64, pub eps: f64, pub t: usize,
    m_a: DVector<f64>, m_b: DVector<f64>, m_w: DMatrix<f64>,
    v_a: DVector<f64>, v_b: DVector<f64>, v_w: DMatrix<f64>,
}
impl Adam {
    pub fn new(nv: usize, nh: usize) -> Self {
        Adam { alpha:0.01, beta1:0.9, beta2:0.999, eps:1e-8, t:0,
            m_a:DVector::zeros(nv), m_b:DVector::zeros(nh), m_w:DMatrix::zeros(nv,nh),
            v_a:DVector::zeros(nv), v_b:DVector::zeros(nh), v_w:DMatrix::zeros(nv,nh) }
    }
    pub fn step(&mut self, rbm: &mut Rbm, g: &VmcGradient) {
        self.t += 1;
        let t = self.t as f64;
        let bc1 = 1. - self.beta1.powf(t);
        let bc2 = 1. - self.beta2.powf(t);
        let alpha = self.alpha;
        let eps   = self.eps;

        self.m_a = self.beta1 * &self.m_a + (1.-self.beta1) * &g.grad_a;
        self.v_a = self.beta2 * &self.v_a + (1.-self.beta2) * g.grad_a.map(|x| x*x);
        let step_a = alpha * (&self.m_a/bc1).zip_map(&(&self.v_a/bc2), |m,v| m/(v.sqrt()+eps));
        rbm.a -= step_a;

        self.m_b = self.beta1 * &self.m_b + (1.-self.beta1) * &g.grad_b;
        self.v_b = self.beta2 * &self.v_b + (1.-self.beta2) * g.grad_b.map(|x| x*x);
        let step_b = alpha * (&self.m_b/bc1).zip_map(&(&self.v_b/bc2), |m,v| m/(v.sqrt()+eps));
        rbm.b -= step_b;

        self.m_w = self.beta1 * &self.m_w + (1.-self.beta1) * &g.grad_w;
        self.v_w = self.beta2 * &self.v_w + (1.-self.beta2) * g.grad_w.map(|x| x*x);
        let step_w = alpha * (&self.m_w/bc1).zip_map(&(&self.v_w/bc2), |m,v| m/(v.sqrt()+eps));
        rbm.w -= step_w;
    }
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

pub struct VmcConfig {
    pub n: usize, pub m: usize, pub j: f64, pub h: f64,
    pub n_samples: usize, pub n_iter: usize,
    pub n_thermalize: usize, pub learning_rate: f64,
}
impl Default for VmcConfig {
    fn default() -> Self {
        VmcConfig { n:8, m:8, j:1., h:1., n_samples:300,
                    n_iter:1000, n_thermalize:500, learning_rate:0.01 }
    }
}

pub struct VmcResult {
    pub energies: Vec<f64>,
    pub final_energy: f64,
    pub acceptance_rate: f64,
}

pub fn run_vmc(cfg: &VmcConfig, rng: &mut impl Rng) -> VmcResult {
    let mut rbm = Rbm::new_random(cfg.n, cfg.m, 0.01, rng);
    let mut sampler = MetropolisSampler::new(cfg.n, &rbm, rng);
    let mut adam = Adam::new(cfg.n, cfg.m);
    adam.alpha = cfg.learning_rate;

    sampler.thermalize(cfg.n_thermalize, &rbm, rng);
    let mut energies = Vec::with_capacity(cfg.n_iter);

    for iter in 0..cfg.n_iter {
        let samples = sampler.collect_samples(cfg.n_samples, &rbm, rng);
        let (e_mean, grad) = estimate_gradient(&samples, &rbm, cfg.j, cfg.h);
        adam.step(&mut rbm, &grad);
        energies.push(e_mean);
        if iter % 100 == 0 || iter == cfg.n_iter-1 {
            println!("  iter {:>4}  E = {:>10.6}  accept = {:.3}",
                iter, e_mean, sampler.acceptance_rate());
        }
    }

    let final_energy = *energies.last().unwrap();
    VmcResult { energies, final_energy, acceptance_rate: sampler.acceptance_rate() }
}

pub fn write_energy_csv(energies: &[f64], path: &str) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(path)?;
    wtr.write_record(["iteration","energy"])?;
    for (i,&e) in energies.iter().enumerate() {
        wtr.write_record(&[i.to_string(), format!("{:.10}",e)])?;
    }
    wtr.flush()?; Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    // Flat wavefunction (all weights zero): psi ratio = 1 for every flip,
    // so off-diagonal term = -h * N.  Full E_loc = -J*(N-1) + (-h*N).
    #[test]
    fn test_local_energy_flat_wavefunction() {
        let n=6; let m=4;
        let rbm = Rbm { a:DVector::zeros(n), b:DVector::zeros(m),
            w:DMatrix::zeros(n,m), n_visible:n, n_hidden:m };
        let spins = vec![1i8; n];
        let theta = rbm.theta(&spins);
        let eloc = local_energy(&spins, &theta, &rbm, 1.0, 0.5);
        let expected = -(n as f64-1.) + (-0.5 * n as f64);
        assert!((eloc-expected).abs()<1e-10, "got {eloc:.8} expected {expected:.8}");
    }

    // Adam drives a simple quadratic toward its minimum.
    #[test]
    fn test_adam_decreases_loss() {
        let mut rbm = Rbm { a:DVector::from_element(1,2.0_f64),
            b:DVector::zeros(1), w:DMatrix::zeros(1,1),
            n_visible:1, n_hidden:1 };
        let mut adam = Adam::new(1,1);
        let initial = rbm.a[0];
        for _ in 0..200 {
            let g = VmcGradient {
                grad_a: DVector::from_element(1, 2.*rbm.a[0]),
                grad_b: DVector::zeros(1), grad_w: DMatrix::zeros(1,1),
            };
            adam.step(&mut rbm, &g);
        }
        assert!(rbm.a[0].abs() < initial,
            "Adam should reduce |a[0]|: started {initial:.4}, got {:.4}", rbm.a[0]);
    }

    // EXIT CRITERION: VMC energy for N=8 converges within 1% of ED reference.
    #[test]
    fn test_vmc_n8_convergence() {
        let mut rng = StdRng::seed_from_u64(42);
        let cfg = VmcConfig::default();
        println!("\nVMC convergence test (N=8, 1000 iterations):");
        let result = run_vmc(&cfg, &mut rng);
        let reference = -9.837951448_f64;
        let rel_err = (result.final_energy - reference).abs() / reference.abs();
        println!("  Final energy:   {:.6}", result.final_energy);
        println!("  ED reference:   {:.6}", reference);
        println!("  Relative error: {:.4}%", rel_err*100.);
        assert!(rel_err < 0.01,
            "relative error {:.4}% exceeds 1% threshold", rel_err*100.);
    }
}
