from flask import Blueprint, jsonify, request
from auto_mpg import DataLoader, DataCleaner, FeatureSelector

select_bp = Blueprint('select', __name__)

@select_bp.route('/select', methods=['POST'])
def select():
    data = request.get_json() or {}
    p = data.get("p", 0.75)

    loader = DataLoader()
    df = loader.load()

    cleaner = DataCleaner(df)
    df = cleaner.fix_missing_numerical()

    selector = FeatureSelector(df, target='mpg')
    selector.apply_variance_threshold(p=p)
    df_selected = selector.get_dataset()

    return jsonify({
        "status": "ok",
        "shape": list(df_selected.shape),
        "columns": list(df_selected.columns)
    })