from flask import Blueprint, jsonify
from auto_mpg import DataLoader, DataCleaner, FeatureSelector, Preprocessor
from auto_mpg.split import Split

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

    X = df.drop(columns=['mpg'])
    y = df['mpg']

    splitter = Split()
    X_train, X_test, y_train, y_test = splitter.split(X, y)

    preprocessor = Preprocessor()
    X_train_scaled, X_test_scaled = preprocessor.scale(X_train, X_test)

    return jsonify({
        "status": "ok",
        "shape_train": list(X_train_scaled.shape),
        "shape_test": list(X_test_scaled.shape)
    })