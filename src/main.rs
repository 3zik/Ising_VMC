use rand::Rng;
pub mod ed;
pub mod rbm;
pub mod sampler;
pub mod vmc;

#[derive(Debug, Clone, PartialEq)]
pub struct SpinConfig {
    pub spins: Vec<i8>,
}
impl SpinConfig {
    pub fn new(spins: Vec<i8>) -> Self {
        assert!(spins.iter().all(|&s| s == 1 || s == -1));
        SpinConfig { spins }
    }
    pub fn random_init(n: usize, rng: &mut impl Rng) -> Self {
        let spins = (0..n).map(|_| if rng.gen_bool(0.5) { 1i8 } else { -1i8 }).collect();
        SpinConfig { spins }
    }
    pub fn len(&self) -> usize { self.spins.len() }
    pub fn flip(&mut self, i: usize) { self.spins[i] *= -1; }
    pub fn energy_ising(&self, j: f64) -> f64 {
        let bond_sum: i32 = (0..self.spins.len()-1)
            .map(|i| self.spins[i] as i32 * self.spins[i+1] as i32).sum();
        -j * bond_sum as f64
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    // ED benchmark
    let n_values = vec![4, 6, 8, 10];
    let h_over_j = vec![0.5, 1.0, 1.5];
    println!("Running ED benchmark grid...");
    let results = ed::run_benchmark_grid(&n_values, &h_over_j, 1.0);
    println!("{:<6} {:>8} {:>16} {:>14}", "N", "h/J", "E0", "E0/site");
    println!("{}", "-".repeat(48));
    for r in &results {
        println!("{:<6} {:>8.2} {:>16.8} {:>14.8}", r.n, r.h_over_j, r.e0, r.e0_per_site);
    }
    std::fs::create_dir_all("results").unwrap();
    ed::write_benchmark_csv(&results, "results/ed_benchmark.csv").unwrap();

    // VMC run
    println!("\nRunning VMC (N=8, J=1, h=1, 1000 iterations)...\n");
    let cfg = vmc::VmcConfig::default();
    let result = vmc::run_vmc(&cfg, &mut rng);

    let ed_ref = -9.837951448_f64;
    let rel_err = (result.final_energy - ed_ref).abs() / ed_ref.abs() * 100.0;
    println!("\nFinal VMC energy : {:.6}", result.final_energy);
    println!("ED reference     : {:.6}", ed_ref);
    println!("Relative error   : {:.4}%", rel_err);
    println!("Acceptance rate  : {:.3}", result.acceptance_rate);

    vmc::write_energy_csv(&result.energies, "results/vmc_energy_N8.csv").unwrap();
    println!("Energy trace written to results/vmc_energy_N8.csv");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    #[test]
    fn test_double_flip_identity() {
        let mut rng = StdRng::seed_from_u64(42);
        let orig = SpinConfig::random_init(8, &mut rng);
        let mut c = orig.clone();
        for i in 0..8 { c.flip(i); c.flip(i); }
        assert_eq!(c, orig);
    }
    #[test]
    fn test_energy_all_up() {
        let n = 8;
        let c = SpinConfig::new(vec![1i8; n]);
        assert!((c.energy_ising(1.0) - (-(n as f64 - 1.0))).abs() < 1e-10);
    }
    #[test]
    fn test_energy_alternating() {
        let n = 8;
        let s: Vec<i8> = (0..n).map(|i| if i%2==0 {1} else {-1}).collect();
        assert!((SpinConfig::new(s).energy_ising(1.0) - (n as f64 - 1.0)).abs() < 1e-10);
    }
    #[test]
    fn test_random_init_values() {
        let mut rng = StdRng::seed_from_u64(99);
        let c = SpinConfig::random_init(100, &mut rng);
        assert!(c.spins.iter().all(|&s| s==1||s==-1));
    }
    #[test]
    fn test_flip_changes_energy() {
        let mut c = SpinConfig::new(vec![1i8; 8]);
        let before = c.energy_ising(1.0);
        c.flip(0);
        assert!((c.energy_ising(1.0) - before - 2.0).abs() < 1e-10);
    }
}
