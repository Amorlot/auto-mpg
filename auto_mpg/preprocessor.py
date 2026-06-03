import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self, df: pd.DataFrame = None, target: str = 'mpg'):
        self.target = target
        self.scaler = StandardScaler()
        if df is not None:
            self.y = df[target]
            self.X = df.drop(columns=[target])

    def standardize(self):
        print("\n--- STANDARDIZZAZIONE (Z-Score) ---")
        X_scaled = self.scaler.fit_transform(self.X)
        self.X = pd.DataFrame(X_scaled, columns=self.X.columns)
        print(f"Media dopo scaling (deve essere ~0):")
        print(self.X.mean().round(4))
        print(f"\nStd dopo scaling (deve essere ~1):")
        print(self.X.std().round(4))
        return self.X

    def scale(self, X_train, X_test):
        # fit solo su train, transform su entrambi
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def get_dataset(self):
        df = self.X.copy()
        df[self.target] = self.y.values
        return df