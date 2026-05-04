use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub mod ed;
pub mod rbm;
pub mod sampler;
pub mod vmc;

#[derive(Debug, Clone, PartialEq)]
pub struct SpinConfig { pub spins: Vec<i8> }
impl SpinConfig {
    pub fn new(spins: Vec<i8>) -> Self { assert!(spins.iter().all(|&s|s==1||s==-1)); SpinConfig{spins} }
    pub fn random_init(n: usize, rng: &mut impl Rng) -> Self {
        SpinConfig { spins: (0..n).map(|_| if rng.gen_bool(0.5){1i8}else{-1i8}).collect() }
    }
    pub fn len(&self) -> usize { self.spins.len() }
    pub fn flip(&mut self, i: usize) { self.spins[i] *= -1; }
    pub fn energy_ising(&self, j: f64) -> f64 {
        -j * (0..self.spins.len()-1).map(|i| self.spins[i] as i32 * self.spins[i+1] as i32).sum::<i32>() as f64
    }
}

fn main() {
    // Read N, h/J, n_iter from command-line args so Python can drive runs
    let args: Vec<String> = std::env::args().collect();
    let n:      usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let hj:     f64   = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let n_iter: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1500);
    let m:      usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(n);
    let seed:   u64   = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(42);

    std::fs::create_dir_all("results").unwrap();

    let j = 1.0_f64;
    let mut rng = StdRng::seed_from_u64(seed);
    let cfg = vmc::VmcConfig {
        n, m, j, h: hj * j,
        n_samples: 300, n_iter,
        n_thermalize: 500, learning_rate: 0.01,
    };

    let result = vmc::run_vmc(&cfg, &mut rng);
    let trace_path = format!("results/energy_N{n}_h{hj:.1}_M{m}.csv");
    vmc::write_energy_csv(&result.energies, &trace_path).unwrap();

    // Print final energy as a single line for Python to parse
    println!("RESULT N={n} h/J={hj:.1} M={m} energy={:.10} accept={:.4}",
        result.final_energy, result.acceptance_rate);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    #[test] fn test_double_flip_identity() {
        let mut rng = StdRng::seed_from_u64(42);
        let orig = SpinConfig::random_init(8, &mut rng);
        let mut c = orig.clone();
        for i in 0..8 { c.flip(i); c.flip(i); }
        assert_eq!(c, orig);
    }
    #[test] fn test_energy_all_up() {
        let c = SpinConfig::new(vec![1i8;8]);
        assert!((c.energy_ising(1.0)+7.).abs()<1e-10);
    }
    #[test] fn test_energy_alternating() {
        let s: Vec<i8>=(0..8usize).map(|i|if i%2==0{1}else{-1}).collect();
        assert!((SpinConfig::new(s).energy_ising(1.0)-7.).abs()<1e-10);
    }
    #[test] fn test_random_init_values() {
        let mut rng = StdRng::seed_from_u64(99);
        assert!(SpinConfig::random_init(100,&mut rng).spins.iter().all(|&s|s==1||s==-1));
    }
    #[test] fn test_flip_changes_energy() {
        let mut c = SpinConfig::new(vec![1i8;8]);
        let before = c.energy_ising(1.0); c.flip(0);
        assert!((c.energy_ising(1.0)-before-2.).abs()<1e-10);
    }
}
