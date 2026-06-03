from flask import Flask
from routes import load_bp, clean_bp, select_bp, standardize_bp, pipeline_bp, eda_bp

app = Flask(__name__)

app.register_blueprint(load_bp)
app.register_blueprint(clean_bp)
app.register_blueprint(select_bp)
app.register_blueprint(standardize_bp)
app.register_blueprint(pipeline_bp)
app.register_blueprint(eda_bp)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)