from flask import Blueprint, jsonify, request
from auto_mpg import DataLoader, DataCleaner
from auto_mpg.eda import Eda

eda_bp = Blueprint('eda', __name__)


def get_df():
    loader = DataLoader()
    df = loader.load()
    cleaner = DataCleaner(df)
    df = cleaner.fix_missing_numerical()
    return df


@eda_bp.route('/eda/info', methods=['GET'])
def info():
    eda = Eda(get_df())
    return jsonify(eda.info())


@eda_bp.route('/eda/qqplot/<colonna>', methods=['GET'])
def qqplot(colonna):
    eda = Eda(get_df())
    eda.qqplot(colonna)
    return jsonify({"status": "ok", "file": f"output_plots/qqplot_{colonna}.png"})


@eda_bp.route('/eda/boxplot/<colonna>', methods=['GET'])
def boxplot(colonna):
    eda = Eda(get_df())
    eda.boxplot(colonna)
    return jsonify({"status": "ok", "file": f"output_plots/boxplot_{colonna}.png"})


@eda_bp.route('/eda/histograms', methods=['GET'])
def histograms():
    eda = Eda(get_df())
    eda.plot_histograms()
    return jsonify({"status": "ok", "file": "output_plots/histograms.png"})


@eda_bp.route('/eda/jarque_bera/<colonna>', methods=['GET'])
def jarque_bera(colonna):
    eda = Eda(get_df())
    return jsonify(eda.jarque_bera_test(colonna))


@eda_bp.route('/eda/normal_test/<colonna>', methods=['GET'])
def normal_test(colonna):
    eda = Eda(get_df())
    return jsonify(eda.normal_test(colonna))


@eda_bp.route('/eda/outliers/<colonna>', methods=['GET'])
def outliers(colonna):
    eda = Eda(get_df())
    return jsonify({"colonna": colonna, "outliers": eda.find_outliers(colonna)})