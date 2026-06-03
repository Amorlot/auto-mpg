import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self, df: pd.DataFrame, target: str = 'mpg'):
        self.target = target
        self.y = df[target]
        self.X = df.drop(columns=[target])
        self.scaler = StandardScaler()

    def standardize(self):
        print("\n--- STANDARDIZZAZIONE (Z-Score) ---")

        X_scaled = self.scaler.fit_transform(self.X)
        self.X = pd.DataFrame(X_scaled, columns=self.X.columns)

        print(f"Media dopo scaling (deve essere ~0):")
        print(self.X.mean().round(4))
        print(f"\nStd dopo scaling (deve essere ~1):")
        print(self.X.std().round(4))

        return self.X

    def get_dataset(self):
        df = self.X.copy()
        df[self.target] = self.y.values
        return df