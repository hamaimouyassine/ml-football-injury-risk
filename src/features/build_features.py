import pandas as pd


def create_reinjury_target(df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    """
    Create reinjury target based on time window.
    """

    df = df.sort_values(["Name", "Date of Injury"]).copy()

    df["reinjury_90"] = 0

    for player in df["Name"].unique():
        player_df = df[df["Name"] == player]

        for i in range(len(player_df) - 1):
            current_idx = player_df.index[i]
            next_idx = player_df.index[i + 1]

            return_date = df.loc[current_idx, "Date of return"]
            next_injury_date = df.loc[next_idx, "Date of Injury"]

            delta = (next_injury_date - return_date).days

            if window_days >= delta >= 0:
                df.loc[current_idx, "reinjury_90"] = 1

    return df
def add_history_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(["Name", "Date of Injury"]).copy()

    df["previous_injuries"] = 0
    df["days_since_last_injury"] = -1

    for player in df["Name"].unique():
        player_df = df[df["Name"] == player]

        for i in range(len(player_df)):
            idx = player_df.index[i]

            # previous injuries
            df.loc[idx, "previous_injuries"] = i

            if i > 0:
                prev_idx = player_df.index[i - 1]
                prev_return = df.loc[prev_idx, "Date of return"]
                current_injury = df.loc[idx, "Date of Injury"]

                delta = (current_injury - prev_return).days
                df.loc[idx, "days_since_last_injury"] = delta

    return df
def select_model_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["days_since_last_injury"] = df["days_since_last_injury"].replace(-1, 999)

    selected_columns = [
        "Age",
        "Age_squared",
        "FIFA rating",
        "absence_days",
        "serious_injury",
        "previous_injuries",
        "days_since_last_injury",
        "risk_ratio",
        "Position",
        "Injury",
        "reinjury_90"
    ]

    return df[selected_columns]

def add_severity_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["serious_injury"] = (df["absence_days"] > 60).astype(int)
    return df

def add_risk_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_ratio"] = df["absence_days"] / (df["days_since_last_injury"] + 1)
    return df
