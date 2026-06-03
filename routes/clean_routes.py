from flask import Blueprint, jsonify
from auto_mpg import DataLoader, DataCleaner

clean_bp = Blueprint('clean', __name__)

@clean_bp.route('/clean', methods=['POST'])
def clean():
    loader = DataLoader()
    df = loader.load()

    cleaner = DataCleaner(df)
    df_clean = cleaner.fix_missing_numerical()

    return jsonify({
        "status": "ok",
        "shape": list(df_clean.shape),
        "missing_rimasti": int(df_clean.isnull().sum().sum())
    })