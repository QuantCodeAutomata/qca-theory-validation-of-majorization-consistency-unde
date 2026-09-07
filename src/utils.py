import numpy as np
from typing import List
import pandas as pd


def simplex_vertices(n: int) -> List[np.ndarray]:
    """Return the n vertices of the n-simplex Delta_n."""
    return [np.eye(n)[i] for i in range(n)]


def format_weights(w: np.ndarray, decimals: int = 4) -> str:
    """Format portfolio weights as a string."""
    return '(' + ', '.join(f'{x:.{decimals}f}' for x in w) + ')'


def print_table(data: dict, title: str = '', decimals: int = 4) -> pd.DataFrame:
    """Create and print a formatted pandas DataFrame from a dict."""
    df = pd.DataFrame(data)
    if title:
        print(f'\n=== {title} ===')
    pd.set_option('display.float_format', f'{{:.{decimals}f}}'.format)
    print(df.to_string(index=False))
    pd.reset_option('display.float_format')
    return df


def validate_weights(w: np.ndarray, tol: float = 1e-6) -> bool:
    """Validate that weights form a valid simplex element."""
    return bool(np.all(w >= -tol) and abs(np.sum(w) - 1.0) < tol)
