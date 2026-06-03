from auto_mpg import DataLoader, DataCleaner, FeatureSelector, Preprocessor

loader = DataLoader()
df = loader.load()
loader.report_missing()

cleaner = DataCleaner(df)
df = cleaner.fix_missing_numerical()

selector = FeatureSelector(df, target='mpg')
selector.apply_variance_threshold(p=0.75)
df = selector.get_dataset()

preprocessor = Preprocessor(df, target='mpg')
X_scaled = preprocessor.standardize()

print(X_scaled.head())