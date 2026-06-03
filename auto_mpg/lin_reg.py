from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np


class LinReg:

    def __init__(self, model_type="linear", alpha=1.0):
        """
        model_type:
            - linear
            - ridge
            - lasso
        """

        self.model_type = model_type
        self.alpha = alpha
        self.model = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def fit(self, X_train, y_train):
        """
        Addestra il modello.
        """

        self.X_train = X_train
        self.y_train = y_train

        if self.model_type == "linear":
            self.model = LinearRegression()

        elif self.model_type == "ridge":
            self.model = Ridge(alpha=self.alpha)

        elif self.model_type == "lasso":
            self.model = Lasso(alpha=self.alpha)

        else:
            raise ValueError(
                "model_type deve essere 'linear', 'ridge' oppure 'lasso'"
            )

        self.model.fit(X_train, y_train)

        return self

    def evaluate(self, X_test, y_test):
        """
        Valuta il modello sul test set.
        """

        self.X_test = X_test
        self.y_test = y_test

        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        n = len(y_test)
        p = X_test.shape[1]

        r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        metrics = {
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "R2_ADJ": r2_adj
        }

        return metrics


    def get_coefficients(self):
        """
        Restituisce i coefficienti del modello.
        """

        return pd.DataFrame({
            "Coefficient": self.model.coef_
        })

    def intercept(self):
        """
        Restituisce l'intercetta.
        """

        return self.model.intercept_