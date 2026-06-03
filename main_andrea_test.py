from auto_mpg import DataLoader, DataCleaner, FeatureSelector

loader = DataLoader()
df = loader.load()
loader.report_missing()

cleaner = DataCleaner(df)
df = cleaner.fix_missing_numerical()

selector = FeatureSelector(df, target='mpg')
X = selector.apply_variance_threshold(p=0.75)

print(X.head())
print(X.shape)