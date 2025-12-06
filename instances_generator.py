import numpy as np

class SetCoverGenerator:

    def __init__(self, n_rows=50, n_cols=100, density=0.4):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.density = density

    def generate(self):
        A = np.random.choice(
            [0, 1],
            size=(self.n_rows, self.n_cols),
            p=[1 - self.density, self.density]
        )

        for i in range(self.n_rows):
            if A[i].sum() < 2:
                cols = np.random.choice(self.n_cols, size=2, replace=False)
                A[i, cols] = 1

        for j in range(self.n_cols):
            if A[:, j].sum() < 2:
                rows = np.random.choice(self.n_rows, size=2, replace=False)
                A[rows, j] = 1

        c = np.ones(self.n_cols, dtype=float)
        b = np.ones(self.n_rows, dtype=float)

        return {'A': A, 'c': c, 'b': b, 'type': 'cover'}



class RandomMILPGenerator:

    def __init__(self, n_cons=4, n_vars=8, density=0.5, max_coef=5):
        self.n_cons = n_cons
        self.n_vars = n_vars
        self.density = density
        self.max_coef = max_coef

    def generate(self):
        A = np.zeros((self.n_cons, self.n_vars), dtype=float)
        mask = np.random.rand(self.n_cons, self.n_vars) < self.density
        A[mask] = np.random.randint(1, self.max_coef + 1, size=mask.sum())

   
        # make sure no collumns or lign is full null
        for i in range(self.n_cons):
            if np.all(A[i] == 0):
                j = np.random.randint(0, self.n_vars)
                A[i, j] = np.random.randint(1, self.max_coef + 1)

        for j in range(self.n_vars):
            if np.all(A[:, j] == 0):
                i = np.random.randint(0, self.n_cons)
                A[i, j] = np.random.randint(1, self.max_coef + 1)

        row_sum = A.sum(axis=1)
        b = np.zeros(self.n_cons, dtype=float)
        for i in range(self.n_cons):
            rs = row_sum[i]
            low = max(1, int(0.4 * rs))
            high = max(low, int(0.6 * rs))
            b[i] = np.random.randint(low, high + 1)

        c = np.random.randint(1, self.max_coef + 1, size=self.n_vars).astype(float)


        return {"A": A, "c": c, "b": b, "type": "cover"}