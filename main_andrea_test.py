from auto_mpg import DataLoader, DataCleaner

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load()
    loader.report_missing()

    cleaner = DataCleaner(df)
    df = cleaner.fix_missing_numerical()
    cleaner.report()

    print(df.head())
    print(df.shape)