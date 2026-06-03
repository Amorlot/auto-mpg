from flask import Blueprint, jsonify
from auto_mpg import DataLoader, DataCleaner, FeatureSelector, Preprocessor

standardize_bp = Blueprint('standardize', __name__)

@standardize_bp.route('/standardize', methods=['POST'])
def standardize():
    loader = DataLoader()
    df = loader.load()

    cleaner = DataCleaner(df)
    df = cleaner.fix_missing_numerical()

    selector = FeatureSelector(df, target='mpg')
    selector.apply_variance_threshold(p=0.75)
    df = selector.get_dataset()

    preprocessor = Preprocessor(df, target='mpg')
    X_scaled = preprocessor.standardize()

    return jsonify({
        "status": "ok",
        "shape": list(X_scaled.shape),
        "sample": X_scaled.head(3).to_dict()
    })