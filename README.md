# Portfolio Diversification Theory: Majorization Consistency Validation

This repository implements and numerically validates theoretical results on portfolio diversification under dependence ambiguity, covering:

- Majorization consistency of portfolio risk functionals (Theorem 1)
- Tail-risk concentration under full dependence ambiguity (Section 7.1)
- Worst-case mean-SD optimization with heterogeneous marginals (Section 7.2)
- Value of dependence information in mean-SD portfolio selection (Section 7.3)
- Tail-shape sensitivity of concentrated tail-risk portfolios (Section 7.4)
- Counterexample: diversification under worst-case SD (Example 1)
- Structured uncertainty reductions (Propositions 6 & 7)
- Weighted robustness under reference and worst-case dependence (Section 8)

## Repository Structure

```
repo/
├── data/           # Synthetic data generation utilities
├── src/            # Core implementations
│   ├── risk_measures.py    # VaR, ES, RVaR, SD
│   ├── majorization.py     # Majorization order and Birkhoff decomposition
│   ├── marginals.py        # Section 7 asset marginals
│   ├── optimization.py     # Portfolio optimization (cvxpy)
│   └── utils.py            # Utilities
├── exp/            # Experiment scripts
├── tests/          # pytest tests
├── results/        # Output artifacts and RESULTS.md
├── README.md
└── .gitignore
```

## Usage

```bash
pip install -r requirements.txt
# Run all experiments
python exp/run_all.py
# Run tests
pytest tests/ -v
```

## Key Dependencies
- numpy, scipy, cvxpy, pandas, matplotlib
