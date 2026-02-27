import pandas as pd

def main():
    df = pd.read_csv("data/raw/reviews.csv")

    print("Total rows:", len(df))
    print("\\nMissing values:")
    print(df.isnull().sum())

    print("\\nClass distribution:")
    print(df["sentiment"].value_counts())

if __name__ == "__main__":
    main()
