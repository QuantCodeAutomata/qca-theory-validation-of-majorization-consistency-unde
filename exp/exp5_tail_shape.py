import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, integrate
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk_measures import normal_var, normal_rvar
from src.marginals import MEANS, SDS, asset3_params, asset3_var, asset3_rvar
from src.utils import print_table

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Fixed asset 1 and 2 risk values
VAR_P = 0.99
RVAR_BETA, RVAR_ALPHA = 0.01, 0.10
R1_VAR = normal_var(4.0, 2.6, VAR_P)
R2_VAR = normal_var(4.8, 0.9, VAR_P)
R1_RVAR = normal_rvar(4.0, 2.6, RVAR_BETA, RVAR_ALPHA)
R2_RVAR = normal_rvar(4.8, 0.9, RVAR_BETA, RVAR_ALPHA)
M1, M2, M3 = 4.0, 4.8, 5.3


def validate_asset3_moments(eta_values):
    """Verify mean and SD invariance for all eta."""
    for eta in eta_values:
        a, b = asset3_params(eta)
        import math
        mean3 = a + b * math.exp(eta**2 / 2)
        var3 = b**2 * (math.exp(2*eta**2) - math.exp(eta**2))
        sd3 = math.sqrt(var3)
        assert abs(mean3 - 5.3) < 1e-6, f'Mean invariance failed at eta={eta}: got {mean3}'
        assert abs(sd3 - 0.2) < 1e-6, f'SD invariance failed at eta={eta}: got {sd3}'
    print('Asset 3 mean/SD invariance validated for all eta values.')


def compute_switching_threshold(m_lo: float, m_hi: float, r_lo: float, r_hi: float) -> float:
    """kappa switching threshold between assets with means m_lo, m_hi and risks r_lo, r_hi.
    kappa* = (m_hi - m_lo) / (r_lo - r_hi) (positive when r_lo > r_hi).
    """
    denom = r_lo - r_hi
    if abs(denom) < 1e-12:
        return np.inf
    return (m_hi - m_lo) / denom


def compute_table6(eta_values) -> pd.DataFrame:
    """Reproduce Table 6: sensitivity to tail shape parameter eta."""
    records = []
    for eta in eta_values:
        r3_var = asset3_var(VAR_P, eta)
        r3_rvar = asset3_rvar(RVAR_BETA, RVAR_ALPHA, eta)

        kappa_23_var = compute_switching_threshold(M2, M3, R2_VAR, r3_var)
        kappa_23_rvar = compute_switching_threshold(M2, M3, R2_RVAR, r3_rvar)

        records.append({
            'eta': round(eta, 2),
            'VaR_0.99_X3': round(r3_var, 4),
            'kappa_23_VaR': round(kappa_23_var, 4),
            'RVaR_X3': round(r3_rvar, 4),
            'kappa_23_RVaR': round(kappa_23_rvar, 4),
        })

    df = pd.DataFrame(records)
    print_table(df, 'Table 6: Tail-Shape Sensitivity')
    return df


def compute_phase_map(eta_grid: np.ndarray, kappa_grid: np.ndarray, rho: str = 'VaR'):
    """Compute phase map: optimal asset at each (eta, kappa) point."""
    phase = np.zeros((len(kappa_grid), len(eta_grid)), dtype=int)
    for j, eta in enumerate(eta_grid):
        if rho == 'VaR':
            r3 = asset3_var(VAR_P, eta)
            r1, r2 = R1_VAR, R2_VAR
        else:
            r3 = asset3_rvar(RVAR_BETA, RVAR_ALPHA, eta)
            r1, r2 = R1_RVAR, R2_RVAR
        risks = np.array([r1, r2, r3])
        for i, kappa in enumerate(kappa_grid):
            f = -np.array([M1, M2, M3]) - kappa * risks
            phase[i, j] = np.argmax(f)
    return phase


def plot_figure2cd(eta_grid, kappa_grid, phase_var, phase_rvar):
    """Generate Figure 2(c)-(d): phase maps over (eta, kappa)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 2: Phase Maps Over (eta, kappa) Plane', fontsize=14, fontweight='bold')

    cmap = plt.get_cmap('Set1', 3)

    for ax, phase, title in zip(axes, [phase_var, phase_rvar], ['(c) VaR_0.99^+', '(d) RVaR_{0.01,0.10}']):
        im = ax.pcolormesh(eta_grid, kappa_grid, phase, cmap=cmap, vmin=-0.5, vmax=2.5)
        ax.set_xlabel('Shape Parameter eta')
        ax.set_ylabel('Risk Aversion kappa')
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.set_ticklabels(['Asset 1', 'Asset 2', 'Asset 3'])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp5_figure2cd.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure 2(c)-(d) saved to {RESULTS_DIR}/exp5_figure2cd.png')


def main():
    print('\n' + '='*60)
    print('EXPERIMENT 5: Tail-Shape Sensitivity of Concentrated Tail-Risk Portfolios')
    print('='*60)

    eta_table = [0.20, 0.40, 0.55, 0.70, 1.00, 1.30, 1.50]

    validate_asset3_moments(eta_table)

    df_t6 = compute_table6(eta_table)

    print('\nExpected Table 6 (sample):')
    print('eta=1.00: VaR=6.0951, kappa_23_VaR=0.6261, RVaR=5.6327, kappa_23_RVaR=0.8067')
    print('eta=1.50: VaR=5.9617, kappa_23_VaR=0.5365, RVaR=5.5083, kappa_23_RVaR=0.6719')

    # Generate phase maps
    print('\nComputing phase maps...')
    eta_grid = np.linspace(0.20, 1.50, 150)
    kappa_grid = np.linspace(0.0, 2.0, 150)

    phase_var = compute_phase_map(eta_grid, kappa_grid, rho='VaR')
    phase_rvar = compute_phase_map(eta_grid, kappa_grid, rho='RVaR')

    plot_figure2cd(eta_grid, kappa_grid, phase_var, phase_rvar)

    df_t6.to_csv(RESULTS_DIR / 'exp5_table6.csv', index=False)
    return df_t6


if __name__ == '__main__':
    main()
