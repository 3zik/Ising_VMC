use nalgebra::DMatrix;
use csv::Writer;
use std::error::Error;

pub fn build_hamiltonian(n: usize, j: f64, h: f64) -> DMatrix<f64> {
    let dim = 1 << n;
    let mut mat = DMatrix::<f64>::zeros(dim, dim);
    for idx in 0..dim {
        for i in 0..n-1 {
            let si = 1 - 2*((idx>>i)&1) as i32;
            let sj = 1 - 2*((idx>>(i+1))&1) as i32;
            mat[(idx,idx)] += -j*(si*sj) as f64;
        }
        for i in 0..n {
            mat[(idx, idx^(1<<i))] += -h;
        }
    }
    mat
}

pub fn ground_state_energy(n: usize, j: f64, h: f64) -> f64 {
    let eig = build_hamiltonian(n, j, h).symmetric_eigen();
    eig.eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min)
}

pub struct EdResult {
    pub n: usize, pub j: f64, pub h: f64,
    pub h_over_j: f64, pub e0: f64, pub e0_per_site: f64,
}

pub fn run_benchmark_grid(n_values: &[usize], h_over_j_values: &[f64], j: f64) -> Vec<EdResult> {
    let mut results = Vec::new();
    for &n in n_values {
        for &hj in h_over_j_values {
            let h = hj * j;
            let e0 = ground_state_energy(n, j, h);
            results.push(EdResult { n, j, h, h_over_j: hj, e0, e0_per_site: e0/n as f64 });
        }
    }
    results
}

pub fn write_benchmark_csv(results: &[EdResult], path: &str) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(path)?;
    wtr.write_record(["N","J","h","h_over_J","E0","E0_per_site"])?;
    for r in results {
        wtr.write_record(&[r.n.to_string(), format!("{:.4}",r.j),
            format!("{:.4}",r.h), format!("{:.4}",r.h_over_j),
            format!("{:.10}",r.e0), format!("{:.10}",r.e0_per_site)])?;
    }
    wtr.flush()?; Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const TOL: f64 = 1e-6;
    #[test] fn test_ed_n4() { assert!((ground_state_energy(4,1.,1.)+4.758770483).abs()<TOL); }
    #[test] fn test_ed_n6() { assert!((ground_state_energy(6,1.,1.)+7.296229814).abs()<TOL); }
    #[test] fn test_ed_n8_exit_criterion() { assert!((ground_state_energy(8,1.,1.)+9.837951448).abs()<TOL); }
    #[test] fn test_ed_pure_ising() { assert!((ground_state_energy(6,1.,0.)+5.0).abs()<TOL); }
    #[test] fn test_ed_pure_field() { assert!((ground_state_energy(6,0.,2.)+12.0).abs()<TOL); }
    #[test] fn test_hamiltonian_symmetric() {
        let m = build_hamiltonian(4,1.,1.);
        assert!((&m - m.transpose()).amax() < 1e-14);
    }
}
