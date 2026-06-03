from flask import Blueprint, jsonify, request
import pandas as pd
from auto_mpg import DataLoader, DataCleaner, FeatureSelector, Preprocessor
from auto_mpg.split import Split
from auto_mpg.linreg import LinReg

model_bp = Blueprint('model', __name__)

# stato globale
_model = None
_preprocessor = None
_feature_cols = None


def train_pipeline(model_type='linear', alpha=1.0):
    global _model, _preprocessor, _feature_cols

    loader = DataLoader()
    df = loader.load()

    cleaner = DataCleaner(df)
    df = cleaner.fix_missing_numerical()

    selector = FeatureSelector(df, target='mpg')
    selector.apply_variance_threshold(p=0.75)
    df = selector.get_dataset()

    X = df.drop(columns=['mpg'])
    y = df['mpg']
    _feature_cols = list(X.columns)

    splitter = Split()
    X_train, X_test, y_train, y_test = splitter.split(X, y)

    _preprocessor = Preprocessor()
    X_train_scaled, X_test_scaled = _preprocessor.scale(X_train, X_test)

    _model = LinReg(model_type=model_type, alpha=alpha)
    _model.fit(X_train_scaled, y_train)

    metrics = _model.evaluate(X_test_scaled, y_test)
    return metrics


@model_bp.route('/model/train', methods=['POST'])
def train():
    data = request.get_json() or {}
    model_type = data.get('model_type', 'linear')
    alpha = data.get('alpha', 1.0)

    metrics = train_pipeline(model_type=model_type, alpha=alpha)

    return jsonify({
        "status": "ok",
        "model_type": model_type,
        "metriche": metrics
    })


@model_bp.route('/model/predict', methods=['POST'])
def predict():
    if _model is None:
        return jsonify({"error": "chiama prima /model/train"}), 400

    data = request.get_json() or {}
    missing = [col for col in _feature_cols if col not in data]
    if missing:
        return jsonify({"error": f"mancano le feature: {missing}"}), 400

    X_input = pd.DataFrame([data])[_feature_cols]
    X_scaled = _preprocessor.scaler.transform(X_input)
    mpg_previsto = _model.model.predict(X_scaled)[0]

    return jsonify({
        "mpg_previsto": round(float(mpg_previsto), 2)
    })


@model_bp.route('/model/coefficients', methods=['GET'])
def coefficients():
    if _model is None:
        return jsonify({"error": "chiama prima /model/train"}), 400

    coeffs = _model.get_coefficients()
    coeffs.index = _feature_cols

    return jsonify({
        "intercetta": float(_model.intercept()),
        "coefficienti": coeffs['Coefficient'].to_dict()
    })