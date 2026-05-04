"""
plot_results.py
Run this from inside your ising_vmc folder after `cargo run --release`.
Requires: pip install matplotlib numpy
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os, numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def smooth(x, w=30):
    return np.convolve(x, np.ones(w)/w, mode='valid')

os.makedirs('results', exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────

summary  = load_csv('results/vmc_summary.csv')
ed_bm    = load_csv('results/ed_benchmark.csv')
sweep    = load_csv('results/sweep_summary.csv')

n_values  = [8, 10, 12, 14, 16]
hj_values = [0.5, 1.0, 1.5]

COLORS = {
    8:  '#534AB7',
    10: '#0F6E56',
    12: '#D85A30',
    14: '#854F0B',
    16: '#888780',
}
HJ_LABEL = {
    0.5: 'h/J = 0.5  (ferromagnet)',
    1.0: 'h/J = 1.0  (critical point)',
    1.5: 'h/J = 1.5  (paramagnet)',
}

# ── Figure 1: Convergence curves ─────────────────────────────────────────────
# One panel per field strength. Solid = VMC (smoothed), dashed = ED reference.

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

for ax, hj in zip(axes, hj_values):
    for n in n_values:
        fname = f'results/energy_N{n}_h{hj:.1f}_M{n}.csv'
        if not os.path.exists(fname):
            continue
        rows = load_csv(fname)
        E = [float(r['energy']) for r in rows]
        I = [int(r['iteration']) for r in rows]
        w = 30
        sm = smooth(E, w)
        ed_val = next(
            float(r['E0']) for r in ed_bm
            if int(r['N']) == n and abs(float(r['h_over_J']) - hj) < 0.01
        )
        ax.plot(I[w-1:], sm, color=COLORS[n], lw=1.5, label=f'N={n}')
        ax.axhline(ed_val, color=COLORS[n], lw=0.8, ls='--', alpha=0.45)

    ax.set_title(HJ_LABEL[hj], fontsize=9.5)
    ax.set_xlabel('Iteration', fontsize=9)
    if hj == 0.5:
        ax.set_ylabel('Energy $E_0$', fontsize=9)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.2, lw=0.5)

fig.suptitle('VMC convergence (solid) vs ED reference (dashed)', fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig('results/fig1_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figure 1 saved: results/fig1_convergence.png')

# ── Figure 2: Energy comparison + error heatmap ───────────────────────────────
# Left: energy per site vs h/J for all N.
# Right: heatmap of relative error (%) across the full grid.

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for n in n_values:
    sub = sorted(
        [r for r in summary if int(r['N']) == n],
        key=lambda r: float(r['h_over_J'])
    )
    hjs = [float(r['h_over_J']) for r in sub]
    vmc = [float(r['VMC_energy']) / n for r in sub]
    ed  = [float(r['ED_energy'])  / n for r in sub]
    ax.plot(hjs, vmc, 'o-',  color=COLORS[n], lw=1.8, ms=7, label=f'N={n} (VMC)')
    ax.plot(hjs, ed,  's--', color=COLORS[n], lw=1.0, ms=4, alpha=0.5)

ax.axvline(1.0, color='#999', lw=1.0, ls=':', label='critical point')
ax.set_xlabel('h / J', fontsize=11)
ax.set_ylabel('Energy per site  $E_0/N$', fontsize=11)
ax.set_title('VMC (circles) vs exact diagonalization (squares)', fontsize=10)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.2, lw=0.5)

ax = axes[1]
err_mat = np.zeros((len(n_values), len(hj_values)))
for i, n in enumerate(n_values):
    for j, hj in enumerate(hj_values):
        row = next(
            (r for r in summary
             if int(r['N']) == n and abs(float(r['h_over_J']) - hj) < 0.01),
            None
        )
        if row:
            err_mat[i, j] = float(row['rel_error_pct'])

im = ax.imshow(err_mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=8)
ax.set_xticks(range(3))
ax.set_xticklabels([f'h/J = {h}' for h in hj_values], fontsize=10)
ax.set_yticks(range(5))
ax.set_yticklabels([f'N = {n}' for n in n_values], fontsize=10)
ax.set_title('Relative error vs exact diagonalization (%)', fontsize=10)
plt.colorbar(im, ax=ax, label='Error (%)')
for i in range(len(n_values)):
    for j in range(len(hj_values)):
        col = 'white' if err_mat[i, j] > 3.5 else 'black'
        ax.text(j, i, f'{err_mat[i, j]:.2f}%',
                ha='center', va='center', fontsize=10, color=col)

fig.tight_layout()
fig.savefig('results/fig2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figure 2 saved: results/fig2_comparison.png')

# ── Figure 3: Hidden-unit sweep ───────────────────────────────────────────────
# N=12, h/J=1.0 with M = N/2, N, 2N.

fig, ax = plt.subplots(figsize=(7, 4.5))
sweep_colors = ['#B5D4F4', '#378ADD', '#0C447C']

for row, color in zip(sweep, sweep_colors):
    m = int(row['M'])
    fname = f'results/energy_N12_h1.0_M{m}.csv'
    if not os.path.exists(fname):
        continue
    rows2 = load_csv(fname)
    E = [float(r['energy']) for r in rows2]
    I = [int(r['iteration']) for r in rows2]
    w = 30
    sm = smooth(E, w)
    err = float(row['rel_error_pct'])
    ax.plot(I[w-1:], sm, color=color, lw=2.2,
            label=f'M = {m}  (error = {err:.2f}%)')

ed12 = float(sweep[0]['ED_energy'])
ax.axhline(ed12, color='#D85A30', lw=1.5, ls='--',
           label=f'ED reference  ({ed12:.4f})')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Energy $E_0$', fontsize=11)
ax.set_title('Effect of hidden-unit count — N = 12, h/J = 1.0', fontsize=11)
ax.legend(fontsize=9.5)
ax.grid(True, alpha=0.2, lw=0.5)
fig.tight_layout()
fig.savefig('results/fig3_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figure 3 saved: results/fig3_sweep.png')

print('\nAll done. Open the results/ folder to find your figures.')
