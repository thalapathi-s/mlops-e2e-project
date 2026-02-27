import pandas as pd

def main():
    df = pd.read_csv("data/raw/reviews.csv")

    # Basic text normalization (can be extended later)
    df["review_text"] = df["review_text"].str.lower()

    # Save processed data
    df.to_csv("data/processed/clean.csv", index=False)

    print("✅ Preprocessing completed")

if __name__ == "__main__":
    main()
