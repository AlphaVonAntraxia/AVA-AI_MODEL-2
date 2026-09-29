from pathlib import Path
import pandas as pd

UNIVERSE_FILE = Path(__file__).resolve().parent / "sp500.csv"


def load_sp500_universe():
    return pd.read_csv(UNIVERSE_FILE)


def search_sp500(query):
    df = load_sp500_universe()
    query = query.strip().lower()

    if not query:
        return df.copy()

    mask = (
        df["ticker"].str.lower().str.contains(query, regex=False)
        | df["company_name"].str.lower().str.contains(query, regex=False)
    )

    return df.loc[mask].reset_index(drop=True)
