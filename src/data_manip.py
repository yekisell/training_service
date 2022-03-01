import pandas as pd
import click


def read_data(path: str) -> pd.core.frame.DataFrame:
    df = pd.read_csv(path, low_memory=False, index_col="Unnamed: 0")
    df["Date"] = pd.to_datetime(df["Date"])
    click.echo(f"Dataset size: {df.shape}")
    return df


def train_valid_split_ts(
    dataset: pd.core.frame.DataFrame, target: str, valid_weeks: int
) -> tuple[
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame,
    pd.core.series.Series,
    pd.core.series.Series,
]:
    train_df, valid_df = (
        dataset.iloc[:-1115 * 7 * valid_weeks, :],
        dataset.iloc[-1115 * 7 * valid_weeks:, :],
    )

    train_df = train_df.query("Open == 1")
    train_df = train_df.drop(["Date", "Open"], axis=1)
    valid_df = valid_df.query("Open == 1")
    valid_df = valid_df.drop(["Date", "Open"], axis=1)

    X_train, y_train = train_df.drop(target, axis=1), train_df[target]
    X_valid, y_valid = valid_df.drop(target, axis=1), valid_df[target]

    return X_train, X_valid, y_train, y_valid
