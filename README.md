# Auto MPG — Machine Learning Pipeline

Progetto di analisi e predizione del consumo di carburante (`mpg`) sulle auto, basato sul dataset [Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg) di UCI.

---

## Struttura del progetto

```
auto_mpg/
├── auto_mpg/
│   ├── __init__.py
│   ├── data_loader.py       # carica il dataset da UCI
│   ├── data_cleaner.py      # gestisce i valori mancanti
│   ├── feature_selector.py  # variance threshold
│   ├── preprocessor.py      # standardizzazione Z-Score
│   ├── split.py             # split train/test 80/20
│   ├── linreg.py            # regressione lineare, Ridge, Lasso
│   └── eda.py               # analisi esplorativa dei dati
├── routes/
│   ├── __init__.py
│   ├── load_routes.py
│   ├── clean_routes.py
│   ├── select_routes.py
│   ├── standardize_routes.py
│   ├── pipeline_routes.py
│   ├── eda_routes.py
│   └── model_routes.py
├── main.py                  # pipeline completa senza Flask
├── app.py                   # Flask API
├── start.sh                 # avvia main.py + app.py
├── Dockerfile
└── requirements.txt
```

---

## Requisiti

- Docker

---

## Avvio rapido

### 1. Clona il repository

```bash
git clone https://github.com/Amorlot/auto-mpg.git
cd auto-mpg
```

### 2. Builda il container

```bash
docker build -t auto-mpg .
```

### 3. Esegui la pipeline completa

```bash
docker run --name auto-mpg-test auto-mpg python main.py
```

### 4. Avvia il server Flask

```bash
docker run -p 5000:5000 --name auto-mpg-flask auto-mpg
```

---

## API Endpoints

### Health
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/health` | Controlla che il server sia attivo |

```bash
curl http://localhost:5000/health
```

---

### Pipeline step by step
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/load` | Carica il dataset da UCI |
| POST | `/clean` | Gestisce i valori mancanti con mediana |
| POST | `/select` | Variance threshold sulle feature |
| POST | `/standardize` | Standardizzazione Z-Score |
| POST | `/pipeline` | Esegue tutti gli step in un colpo |

```bash
curl -X POST http://localhost:5000/load
curl -X POST http://localhost:5000/clean
curl -X POST http://localhost:5000/select \
  -H "Content-Type: application/json" \
  -d '{"p": 0.75}'
curl -X POST http://localhost:5000/standardize
curl -X POST http://localhost:5000/pipeline
```

---

### EDA — Analisi Esplorativa
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/eda/info` | Statistiche base del dataset |
| GET | `/eda/outliers/<colonna>` | Outlier con metodo IQR |
| GET | `/eda/jarque_bera/<colonna>` | Test di normalità Jarque-Bera |
| GET | `/eda/normal_test/<colonna>` | Test D'Agostino-Pearson |
| GET | `/eda/qqplot/<colonna>` | Genera QQ Plot |
| GET | `/eda/boxplot/<colonna>` | Genera Boxplot |
| GET | `/eda/histograms` | Genera istogrammi di tutte le feature |

```bash
curl http://localhost:5000/eda/info
curl http://localhost:5000/eda/outliers/horsepower
curl http://localhost:5000/eda/jarque_bera/horsepower
curl http://localhost:5000/eda/normal_test/horsepower
curl http://localhost:5000/eda/qqplot/horsepower
curl http://localhost:5000/eda/boxplot/horsepower
curl http://localhost:5000/eda/histograms
```

---

### Modello
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/model/train` | Addestra il modello |
| POST | `/model/predict` | Prevede il consumo mpg |
| GET | `/model/coefficients` | Restituisce i coefficienti |

```bash
# regressione lineare
curl -X POST http://localhost:5000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "linear"}'

# Ridge
curl -X POST http://localhost:5000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "ridge", "alpha": 0.5}'

# Lasso
curl -X POST http://localhost:5000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "lasso", "alpha": 0.5}'

# previsione
curl -X POST http://localhost:5000/model/predict \
  -H "Content-Type: application/json" \
  -d '{"displacement": 307, "cylinders": 8, "horsepower": 130, "weight": 3504, "acceleration": 12, "model_year": 70, "origin": 1}'

# coefficienti
curl http://localhost:5000/model/coefficients
```

---

## Dataset

| Variabile | Tipo | Descrizione |
|-----------|------|-------------|
| `mpg` | Target | Consumo carburante (miglia per gallone) |
| `cylinders` | Feature | Numero di cilindri |
| `displacement` | Feature | Cilindrata |
| `horsepower` | Feature | Potenza (6 valori mancanti) |
| `weight` | Feature | Peso |
| `acceleration` | Feature | Accelerazione |
| `model_year` | Feature | Anno del modello |
| `origin` | Feature | Origine geografica |

---

## Risultati

| Modello | MSE | RMSE | R² |
|---------|-----|------|----|
| Linear | 8.20 | 2.86 | 0.847 |
| Ridge | 8.20 | 2.86 | 0.847 |
| Lasso | 8.15 | 2.85 | 0.848 |

---

## Autori

- **Andrea** — DataLoader, DataCleaner, FeatureSelector, Preprocessor, Flask API
- **Riccardo** — EDA, Split, LinReg
