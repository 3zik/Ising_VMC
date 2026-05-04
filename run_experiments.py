"""
run_experiments.py
Run this from inside your ising_vmc folder BEFORE plot_results.py.
It calls the Rust binary once per experiment and writes all CSVs.

Usage:
    python run_experiments.py
"""

import subprocess, csv, os, re

os.makedirs('results', exist_ok=True)

BINARY = './target/release/ising_vmc'

N_VALUES  = [8, 10, 12, 14, 16]
HJ_VALUES = [0.5, 1.0, 1.5]

# Exact diagonalization reference energies (open BC, J=1)
ED_REF = {
    (8,  0.5): -7.64059255,  (8,  1.0): -9.83795145,  (8,  1.5): -13.19140495,
    (10, 0.5): -9.76550396,  (10, 1.0): -12.38149000,  (10, 1.5): -16.53525495,
    (12, 0.5): -11.89204487, (12, 1.0): -14.92597111,  (12, 1.5): -19.87910704,
    (14, 0.5): -14.01899646, (14, 1.0): -17.47100405,  (14, 1.5): -23.22295943,
    (16, 0.5): -16.14605096, (16, 1.0): -20.01638790,  (16, 1.5): -26.56681187,
}

summary = []
total = len(N_VALUES) * len(HJ_VALUES)
done  = 0

print('=== VMC Experiment Grid ===\n')

for n in N_VALUES:
    for hj in HJ_VALUES:
        done += 1
        m      = n
        n_iter = 1500
        seed   = n * 100 + int(hj * 10)

        print(f'[{done}/{total}]  N={n}  h/J={hj:.1f}  ...', end=' ', flush=True)

        proc = subprocess.run(
            [BINARY, str(n), str(hj), str(n_iter), str(m), str(seed)],
            capture_output=True, text=True, timeout=300
        )

        m_energy = re.search(r'energy=([-\d.]+)', proc.stdout)
        m_accept = re.search(r'accept=([\d.]+)',  proc.stdout)
        vmc_e  = float(m_energy.group(1)) if m_energy else float('nan')
        accept = float(m_accept.group(1)) if m_accept else 0.0
        ref    = ED_REF[(n, hj)]
        err    = abs(vmc_e - ref) / abs(ref) * 100.0

        print(f'VMC={vmc_e:.6f}  ED={ref:.6f}  err={err:.4f}%  accept={accept:.3f}')

        summary.append({
            'N':               n,
            'h_over_J':        hj,
            'ED_energy':       ref,
            'VMC_energy':      vmc_e,
            'rel_error_pct':   err,
            'acceptance_rate': accept,
        })

# ── Hidden-unit sweep: N=12, h/J=1.0, M = 6, 12, 24 ────────────────────────
print('\n=== Hidden-Unit Sweep (N=12, h/J=1.0) ===\n')
sweep = []
ed_ref_12 = ED_REF[(12, 1.0)]

for m in [6, 12, 24]:
    seed = 999 + m
    print(f'  M={m}  ...', end=' ', flush=True)
    proc = subprocess.run(
        [BINARY, '12', '1.0', '1500', str(m), str(seed)],
        capture_output=True, text=True, timeout=300
    )
    m_e   = re.search(r'energy=([-\d.]+)', proc.stdout)
    vmc_e = float(m_e.group(1)) if m_e else float('nan')
    err   = abs(vmc_e - ed_ref_12) / abs(ed_ref_12) * 100.0
    print(f'VMC={vmc_e:.6f}  err={err:.4f}%')
    sweep.append({'M': m, 'VMC_energy': vmc_e,
                  'ED_energy': ed_ref_12, 'rel_error_pct': err})

# ── Write CSVs ───────────────────────────────────────────────────────────────
with open('results/vmc_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)

with open('results/sweep_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['M','VMC_energy','ED_energy','rel_error_pct'])
    w.writeheader(); w.writerows(sweep)

with open('results/ed_benchmark.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['N','J','h','h_over_J','E0','E0_per_site'])
    for (n, hj), e0 in ED_REF.items():
        w.writerow([n, '1.0000', f'{hj:.4f}', f'{hj:.4f}', f'{e0:.10f}', f'{e0/n:.10f}'])

print('\nAll done.')
print('CSVs written to results/')
print('Now run: python plot_results.py')
