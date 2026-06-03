import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


class DataLoader:
    def __init__(self):
        self.df = None

    def load(self):
        auto_mpg = fetch_ucirepo(id=9)
        X = auto_mpg.data.features.copy()
        y = auto_mpg.data.targets.copy()

        self.df = pd.concat([X, y], axis=1)

        # pulizia valori tipo '?' o spazi
        self.df.replace(
            to_replace=r'^\s*(\?|nan|NaN|N/A|NA)\s*$',
            value=np.nan,
            regex=True,
            inplace=True
        )

        # car_name è solo un ID, non serve
        if 'car_name' in self.df.columns:
            self.df.drop(columns=['car_name'], inplace=True)

        # forza horsepower a float
        self.df['horsepower'] = pd.to_numeric(self.df['horsepower'], errors='coerce')

        return self.df

    def report_missing(self):
        print("\nVALORI MANCANTI")
        total = len(self.df)
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            print("Nessun valore mancante.")
            return

        for col, count in missing.items():
            pct = (count / total) * 100
            print(f"{col:<20} | {count} mancanti ({pct:.2f}%)")