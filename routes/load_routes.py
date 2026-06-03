from flask import Blueprint, jsonify
from auto_mpg import DataLoader

load_bp = Blueprint('load', __name__)

@load_bp.route('/load', methods=['POST'])
def load():
    loader = DataLoader()
    df = loader.load()
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()

    return jsonify({
        "status": "ok",
        "shape": list(df.shape),
        "columns": list(df.columns),
        "missing": missing
    })