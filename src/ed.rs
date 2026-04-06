use nalgebra::DMatrix;

/// Build the full 2^N x 2^N Hamiltonian matrix for the 1D TFIM.
///
/// H = -J * sum_{i=0}^{N-2} sz_i * sz_{i+1}   (Ising bonds, open BC)
///   - h * sum_{i=0}^{N-1} sx_i                (transverse field)
///
/// Basis: integer index 0..2^N-1.  Bit i of the index gives spin i:
///   bit = 0  =>  sz = +1  (spin up)
///   bit = 1  =>  sz = -1  (spin down)
///
/// sx_i flips bit i, so it contributes off-diagonal entries of -h.
pub fn build_hamiltonian(n: usize, j: f64, h: f64) -> DMatrix<f64> {
    assert!(n <= 16, "N too large for dense ED (use N <= 16)");
    let dim = 1 << n; // 2^N
    let mut mat = DMatrix::<f64>::zeros(dim, dim);

    for idx in 0..dim {
        // ── diagonal: sz_i * sz_{i+1} bonds ─────────────────────────────
        for i in 0..n - 1 {
            let si = 1 - 2 * ((idx >> i) & 1) as i32;      // +1 or -1
            let sj = 1 - 2 * ((idx >> (i + 1)) & 1) as i32;
            mat[(idx, idx)] += -j * (si * sj) as f64;
        }

        // ── off-diagonal: sx_i flips bit i ──────────────────────────────
        for i in 0..n {
            let flipped = idx ^ (1 << i);
            mat[(idx, flipped)] += -h;
        }
    }
    mat
}

/// Return the ground-state energy of the 1D TFIM via full diagonalization.
/// Uses nalgebra's symmetric eigendecomposition (all eigenvalues computed).
pub fn ground_state_energy(n: usize, j: f64, h: f64) -> f64 {
    let mat = build_hamiltonian(n, j, h);
    let eig = mat.symmetric_eigen();
    eig.eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values computed with NumPy (open BC, bit=0 => sz=+1).
    // N=4: -4.75877048,  N=6: -7.29622981,  N=8: -9.83795145
    const TOL: f64 = 1e-6;

    #[test]
    fn test_ed_n4() {
        let e0 = ground_state_energy(4, 1.0, 1.0);
        let reference = -4.758770483_f64;
        assert!(
            (e0 - reference).abs() < TOL,
            "N=4: got {e0:.9}, expected {reference:.9}"
        );
    }

    #[test]
    fn test_ed_n6() {
        let e0 = ground_state_energy(6, 1.0, 1.0);
        let reference = -7.296229814_f64;
        assert!(
            (e0 - reference).abs() < TOL,
            "N=6: got {e0:.9}, expected {reference:.9}"
        );
    }

    /// Exit criterion: N=8, J=1, h=1 matches reference to within 1e-6.
    #[test]
    fn test_ed_n8_exit_criterion() {
        let e0 = ground_state_energy(8, 1.0, 1.0);
        let reference = -9.837951448_f64;
        assert!(
            (e0 - reference).abs() < TOL,
            "N=8: got {e0:.9}, expected {reference:.9}"
        );
    }

    /// Limiting case: h=0 means all spins align, E0 = -J*(N-1).
    #[test]
    fn test_ed_pure_ising() {
        let n = 6;
        let e0 = ground_state_energy(n, 1.0, 0.0);
        let expected = -(n as f64 - 1.0);
        assert!(
            (e0 - expected).abs() < TOL,
            "pure Ising (h=0): got {e0:.9}, expected {expected:.9}"
        );
    }

    /// Limiting case: J=0 means non-interacting spins in field h.
    /// Each spin contributes -h to the GS energy, so E0 = -h*N.
    #[test]
    fn test_ed_pure_field() {
        let n = 6;
        let h = 2.0;
        let e0 = ground_state_energy(n, 0.0, h);
        let expected = -h * n as f64;
        assert!(
            (e0 - expected).abs() < TOL,
            "pure field (J=0): got {e0:.9}, expected {expected:.9}"
        );
    }

    /// Hamiltonian must be symmetric (necessary condition for correctness).
    #[test]
    fn test_hamiltonian_symmetric() {
        let mat = build_hamiltonian(4, 1.0, 1.0);
        let diff = (&mat - mat.transpose()).amax();
        assert!(diff < 1e-14, "Hamiltonian not symmetric: max asymmetry = {diff}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark grid + CSV export
// ─────────────────────────────────────────────────────────────────────────────

use csv::Writer;
use std::error::Error;

/// One row of the benchmark table.
#[derive(Debug)]
pub struct EdResult {
    pub n: usize,
    pub j: f64,
    pub h: f64,
    pub h_over_j: f64,
    pub e0: f64,
    pub e0_per_site: f64,
}

/// Run ED over a grid of N and h/J values (J fixed at 1.0).
/// Returns every (N, h/J) combination as a flat Vec.
pub fn run_benchmark_grid(
    n_values: &[usize],
    h_over_j_values: &[f64],
    j: f64,
) -> Vec<EdResult> {
    let mut results = Vec::new();
    for &n in n_values {
        for &hj in h_over_j_values {
            let h = hj * j;
            let e0 = ground_state_energy(n, j, h);
            results.push(EdResult {
                n,
                j,
                h,
                h_over_j: hj,
                e0,
                e0_per_site: e0 / n as f64,
            });
        }
    }
    results
}

/// Write benchmark results to a CSV file at `path`.
pub fn write_benchmark_csv(results: &[EdResult], path: &str) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(path)?;
    wtr.write_record(["N", "J", "h", "h_over_J", "E0", "E0_per_site"])?;
    for r in results {
        wtr.write_record(&[
            r.n.to_string(),
            format!("{:.4}", r.j),
            format!("{:.4}", r.h),
            format!("{:.4}", r.h_over_j),
            format!("{:.10}", r.e0),
            format!("{:.10}", r.e0_per_site),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}
