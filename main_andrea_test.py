from auto_mpg import DataLoader, DataCleaner, FeatureSelector, Preprocessor
from auto_mpg.split import Split

# 1. carica
loader = DataLoader()
df = loader.load()
loader.report_missing()

# 2. pulizia
cleaner = DataCleaner(df)
df = cleaner.fix_missing_numerical()

# 3. feature selection
selector = FeatureSelector(df, target='mpg')
selector.apply_variance_threshold(p=0.75)
df = selector.get_dataset()

# 4. split X e y
X = df.drop(columns=['mpg'])
y = df['mpg']

splitter = Split()
X_train, X_test, y_train, y_test = splitter.split(X, y)

# 5. scaling DOPO lo split
preprocessor = Preprocessor()
X_train_scaled, X_test_scaled = preprocessor.scale(X_train, X_test)

print("\n--- RISULTATO FINALE ---")
print(f"X_train: {X_train_scaled.shape}")
print(f"X_test:  {X_test_scaled.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")