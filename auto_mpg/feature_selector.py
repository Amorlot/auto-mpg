import pandas as pd
from sklearn.feature_selection import VarianceThreshold


class FeatureSelector:
    def __init__(self, df: pd.DataFrame, target: str = 'mpg'):
        self.target = target
        self.y = df[target]
        self.X = df.drop(columns=[target])

    def apply_variance_threshold(self, p: float = 0.75):
        soglia = p * (1 - p)

        selector = VarianceThreshold(threshold=soglia)
        X_ridotto = selector.fit_transform(self.X)

        maschera = selector.get_support()
        cols_mantenute = self.X.columns[maschera].tolist()
        cols_rimosse = self.X.columns[~maschera].tolist()

        print(f"\n--- VARIANCE THRESHOLD ---")
        print(f"Feature rimosse: {cols_rimosse}")
        print(f"Feature mantenute: {cols_mantenute}")

        self.X = pd.DataFrame(X_ridotto, columns=cols_mantenute)
        return self.X
    def get_dataset(self):
        df = self.X.copy()
        df[self.target] = self.y.values
        return df