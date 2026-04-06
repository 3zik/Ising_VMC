use rand::Rng;

pub mod ed;
pub mod rbm;
pub mod sampler;

// ---------------------------------------------------------------------------
// SpinConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct SpinConfig {
    pub spins: Vec<i8>,
}

impl SpinConfig {
    pub fn new(spins: Vec<i8>) -> Self {
        assert!(spins.iter().all(|&s| s == 1 || s == -1), "spins must be +-1");
        SpinConfig { spins }
    }

    pub fn random_init(n: usize, rng: &mut impl Rng) -> Self {
        let spins = (0..n).map(|_| if rng.gen_bool(0.5) { 1i8 } else { -1i8 }).collect();
        SpinConfig { spins }
    }

    pub fn len(&self) -> usize {
        self.spins.len()
    }

    pub fn flip(&mut self, i: usize) {
        self.spins[i] *= -1;
    }

    pub fn energy_ising(&self, j: f64) -> f64 {
        let n = self.spins.len();
        let bond_sum: i32 = (0..n - 1)
            .map(|i| self.spins[i] as i32 * self.spins[i + 1] as i32)
            .sum();
        -j * bond_sum as f64
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // ── SpinConfig smoke test ────────────────────────────────────────────
    let mut rng = rand::thread_rng();
    let config = SpinConfig::random_init(8, &mut rng);
    println!("Random spin config (N=8): {:?}", config.spins);
    println!("Ising energy (J=1):       {:.6}\n", config.energy_ising(1.0));

    // ── ED benchmark grid ────────────────────────────────────────────────
    let n_values = vec![4, 6, 8, 10];
    let h_over_j  = vec![0.5, 1.0, 1.5];
    let j = 1.0;

    println!("Running ED benchmark grid...");
    let results = ed::run_benchmark_grid(&n_values, &h_over_j, j);

    // Print table
    println!("\n{:<6} {:>8} {:>8} {:>16} {:>14}", "N", "J", "h/J", "E0", "E0/site");
    println!("{}", "-".repeat(56));
    for r in &results {
        println!(
            "{:<6} {:>8.2} {:>8.2} {:>16.8} {:>14.8}",
            r.n, r.j, r.h_over_j, r.e0, r.e0_per_site
        );
    }

    // Write CSV
    let csv_path = "results/ed_benchmark.csv";
    std::fs::create_dir_all("results").expect("could not create results/");
    ed::write_benchmark_csv(&results, csv_path).expect("CSV write failed");
    println!("\nBenchmark CSV written to {csv_path}");
}

// ---------------------------------------------------------------------------
// SpinConfig tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_double_flip_identity() {
        let mut rng = StdRng::seed_from_u64(42);
        let original = SpinConfig::random_init(8, &mut rng);
        let mut config = original.clone();
        for i in 0..8 { config.flip(i); config.flip(i); }
        assert_eq!(config, original);
    }

    #[test]
    fn test_energy_all_up() {
        let n = 8;
        let config = SpinConfig::new(vec![1i8; n]);
        let expected = -(n as f64 - 1.0);
        assert!((config.energy_ising(1.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_energy_alternating() {
        let n = 8;
        let spins: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let config = SpinConfig::new(spins);
        let expected = n as f64 - 1.0;
        assert!((config.energy_ising(1.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_random_init_values() {
        let mut rng = StdRng::seed_from_u64(99);
        let config = SpinConfig::random_init(100, &mut rng);
        assert!(config.spins.iter().all(|&s| s == 1 || s == -1));
    }

    #[test]
    fn test_flip_changes_energy() {
        let n = 8;
        let j = 1.0_f64;
        let mut config = SpinConfig::new(vec![1i8; n]);
        let e_before = config.energy_ising(j);
        config.flip(0);
        let e_after = config.energy_ising(j);
        assert!((e_after - e_before - 2.0 * j).abs() < 1e-10);
    }
}
