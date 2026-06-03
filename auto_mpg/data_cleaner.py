import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.imputer = SimpleImputer(strategy='median')

    def fix_missing_numerical(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        before = self.df[num_cols].isnull().sum()

        self.df[num_cols] = self.imputer.fit_transform(self.df[num_cols])

        print("\n--- VALORI MANCANTI GESTITI ---")
        for col in num_cols:
            if before[col] > 0:
                idx = list(num_cols).index(col)
                print(f"{col}: {before[col]} mancanti → mediana ({self.imputer.statistics_[idx]:.2f})")

        return self.df

    def report(self):
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("Nessun valore mancante rimasto.")
        else:
            print(missing)