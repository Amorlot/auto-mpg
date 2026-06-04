import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self, df: pd.DataFrame = None, target: str = 'mpg'):
        self.target = target
        self.scaler = StandardScaler()
        if df is not None:
            self.y = df[target]
            self.X = df.drop(columns=[target])

    def scale(self, X_train, X_test):
        # fit solo su train, transform su entrambi
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def get_dataset(self):
        if not hasattr(self, 'X'):
            raise RuntimeError("Preprocessor deve essere inizializzato con un DataFrame: Preprocessor(df)")
        df = self.X.copy()
        df[self.target] = self.y.values
        return df