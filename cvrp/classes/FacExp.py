import itertools
import pandas as pd

class FacExp:
    """
    Factorial experiment manager for 2-level designs with fractional factorial support.
    - grid: dict of {factor_name: [level1, level2]} (order matters, mapped to A, B, C, ...)
    - defining_contrasts: list of strings, each string is a product of main effects (e.g., 'AB', 'BCD')
    - Stores mapping: {'A': key1, 'B': key2, ...} in self.letter_to_factor
    """
    def __init__(self, grid, defining_contrasts=None):
        self.grid = grid
        self.factors = list(grid.keys())
        self.levels = [grid[k] for k in self.factors]
        self.n_factors = len(self.factors)
        self.defining_contrasts = defining_contrasts or []
        # Map A, B, C, ... to user factor names
        self.letters = [chr(ord('A') + i) for i in range(self.n_factors)]
        self.letter_to_factor = {letter: factor for letter, factor in zip(self.letters, self.factors)}
        self.factor_to_letter = {factor: letter for letter, factor in self.letter_to_factor.items()}
        self._build_full_factorial()
        self._apply_defining_contrasts()

    def _build_full_factorial(self):
        # Each row is a dict: {factor: level}
        self.full = [dict(zip(self.factors, prod)) for prod in itertools.product(*self.levels)]

    def _apply_defining_contrasts(self):
        if not self.defining_contrasts:
            self.frac = self.full
            return
        # Map levels to +1/-1 for contrast math
        level_map = {}
        for i, k in enumerate(self.factors):
            vals = self.levels[i]
            if len(vals) != 2:
                raise ValueError(f"Factor {k} does not have 2 levels.")
            level_map[k] = {vals[0]: 1, vals[1]: -1}
        filtered = []
        for row in self.full:
            keep = True
            for contrast in self.defining_contrasts:
                prod = 1
                for c in contrast:
                    # c is a letter (A, B, ...), map to user factor name
                    factor = self.letter_to_factor[c]
                    prod *= level_map[factor][row[factor]]
                if prod != 1:
                    keep = False
                    break
            if keep:
                filtered.append(row)
        self.frac = filtered

    def get_exp_combinations(self):
        """Return a pandas DataFrame of the fractional factorial experiment (levels as original values)."""
        return pd.DataFrame(self.frac)

    def get_exp_binary_table(self):
        """Return a DataFrame with levels as 0 (first level) and 1 (second level) for each factor."""
        rows = []
        for row in self.frac:
            bin_row = {}
            for i, factor in enumerate(self.factors):
                val = row[factor]
                if val == self.levels[i][0]:
                    bin_row[factor] = 0
                elif val == self.levels[i][1]:
                    bin_row[factor] = 1
                else:
                    bin_row[factor] = None
            rows.append(bin_row)
        return pd.DataFrame(rows)

    def get_exp_sign_table(self):
        """Return a DataFrame with levels as '-' (first level) and '+' (second level) for each factor."""
        rows = []
        for row in self.frac:
            sign_row = {}
            for i, factor in enumerate(self.factors):
                val = row[factor]
                if val == self.levels[i][0]:
                    sign_row[factor] = '-'
                elif val == self.levels[i][1]:
                    sign_row[factor] = '+'
                else:
                    sign_row[factor] = '?'
            rows.append(sign_row)
        return pd.DataFrame(rows)
