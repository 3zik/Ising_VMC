# Ising_VMC

A **Variational Monte Carlo (VMC)** ground-state solver for the **1D transverse-field
Ising model (TFIM)**, written in Rust. The many-body wavefunction is represented by a
**Restricted Boltzmann Machine (RBM)** and optimized with Metropolis sampling and Adam.
Results are validated against **exact diagonalization (ED)** on small systems.

The core solver is implemented in Rust (`nalgebra`); a thin Python layer drives parameter
sweeps and plots the results.

---

## The problem

The transverse-field Ising chain (open boundary conditions, `N` sites) has Hamiltonian

```
H = -J * Σ_{i=0}^{N-2} σ^z_i σ^z_{i+1}  -  h * Σ_{i=0}^{N-1} σ^x_i
```

The goal is to find the ground-state energy `E₀`. Diagonalizing `H` directly costs
`2^N × 2^N`, which is only feasible for small `N`. VMC sidesteps this: instead of storing
the full `2^N`-dimensional state, it parameterizes the wavefunction with a compact ansatz
and estimates energies by sampling.

Because every off-diagonal matrix element of `H` is `-h ≤ 0`, the ground state can be taken
real and positive (Perron–Frobenius), so a **real-valued** RBM ansatz is sufficient — no
complex parameters or sign structure are needed.

## The method

**Ansatz (`rbm.rs`).** The (log) wavefunction amplitude for a spin configuration
`s ∈ {−1,+1}^N` is

```
log ψ(s) = Σ_i a_i s_i  +  Σ_j log(2 cosh θ_j),     θ = b + Wᵀ s
```

with visible biases `a`, hidden biases `b`, and weights `W` (`N_hidden = M` hidden units).
`log(2 cosh x)` is evaluated in a numerically stable form. The module also provides the
single-flip log-ratio `log[ψ(sⁱ)/ψ(s)]` and the log-derivatives `O = ∂ log ψ / ∂θ` used by
the sampler and the gradient.

**Sampling (`sampler.rs`).** A single-spin-flip Metropolis sampler proposes flipping a
random site and accepts with probability `min(1, |ψ(sⁱ)/ψ(s)|²) = min(1, exp(2·log_ratio))`.
The pre-activation vector `θ` is cached and updated incrementally on each accepted flip
rather than recomputed from scratch.

**Local energy and gradient (`vmc.rs`).**

```
E_loc(s) = -J Σ_i s_i s_{i+1}              (diagonal Ising term)
           - h Σ_i ψ(sⁱ)/ψ(s)             (off-diagonal transverse field)

∂E/∂θ = 2 ( ⟨E_loc · O⟩ − ⟨E_loc⟩ ⟨O⟩ )
```

The energy is the sample mean of `E_loc`; the gradient is the standard VMC covariance
estimator. Parameters are updated with **Adam** (`β₁=0.9, β₂=0.999, ε=1e-8`).

**Validation (`ed.rs`).** Builds the full `2^N × 2^N` Hamiltonian and finds the ground state
via symmetric eigendecomposition. Used as the source of truth in the test suite and as the
reference for the experiment grid.

---

## Project structure

```
src/
  main.rs      CLI entry point; SpinConfig type; writes per-run energy traces
  rbm.rs       RBM wavefunction: amplitudes, single-flip ratios, log-derivatives
  sampler.rs   Metropolis single-spin-flip sampler with cached θ
  vmc.rs       Local energy, gradient estimator, Adam, training loop
  ed.rs        Exact diagonalization reference (ground-state energy, benchmark grid)
run_experiments.py   Drives the Rust binary over a grid of (N, h/J) and a hidden-unit sweep
plot_results.py      Plots energy convergence, VMC-vs-ED error, and the hidden-unit sweep
```

## Build and run

Requires a recent Rust toolchain.

```bash
cargo build --release
```

Run a single calculation directly. Arguments are
`N  h/J  n_iter  M(hidden units)  seed`:

```bash
# N=8 chain, h/J=1.0, 1500 optimization steps, M=8 hidden units, seed 42
./target/release/ising_vmc 8 1.0 1500 8 42
```

This prints a one-line summary (final energy and acceptance rate) and writes the full
energy-vs-iteration trace to `results/energy_N8_h1.0_M8.csv`.

## Experiments and plots

The Python harness runs the binary across `N ∈ {8,10,12,14,16}`, `h/J ∈ {0.5, 1.0, 1.5}`,
plus a hidden-unit sweep (`M ∈ {6,12,24}` at `N=12`), compares against embedded ED
references, and writes summary CSVs:

```bash
python run_experiments.py     # writes results/*.csv
python plot_results.py         # writes plots to results/  (needs matplotlib, numpy)
```

## Tests

```bash
cargo test
```

The suite covers each layer, including:

- **ED** ground-state energies at `N=4,6,8`, the pure-Ising and pure-field limits, and
  Hamiltonian symmetry.
- **Sampler** invariants: cached `θ` stays in sync with a full recompute, acceptance rate is
  sane after thermalization, and the mean magnetization of a uniform-weight RBM is ~0.
- **VMC** end-to-end: the local energy for a flat wavefunction matches the closed-form value,
  Adam reduces a quadratic loss, and — the key exit criterion — the VMC energy for `N=8`
  converges to **within 1%** of the ED reference `E₀ = −9.83795145`.

## Validation reference

Exact ground-state energies (open BC, `J=1`) used to check the solver:

| N  | h/J = 0.5 | h/J = 1.0  | h/J = 1.5  |
|----|-----------|------------|------------|
| 8  | −7.640593 | −9.837951  | −13.191405 |
| 10 | −9.765504 | −12.381490 | −16.535255 |
| 12 | −11.892045| −14.925971 | −19.879107 |
| 14 | −14.018996| −17.471004 | −23.222959 |
| 16 | −16.146051| −20.016388 | −26.566812 |

## References

- G. Carleo and M. Troyer, *Solving the quantum many-body problem with artificial neural
  networks*, Science **355**, 602 (2017) — the RBM-wavefunction VMC approach.
- D. P. Kingma and J. Ba, *Adam: A Method for Stochastic Optimization* (2015).
