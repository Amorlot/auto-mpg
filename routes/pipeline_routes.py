from flask import Blueprint, jsonify
from auto_mpg import DataLoader, DataCleaner, FeatureSelector, Preprocessor

pipeline_bp = Blueprint('pipeline', __name__)

@pipeline_bp.route('/pipeline', methods=['POST'])
def pipeline():
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
        "columns": list(X_scaled.columns),
        "sample": X_scaled.head(3).to_dict()
    })