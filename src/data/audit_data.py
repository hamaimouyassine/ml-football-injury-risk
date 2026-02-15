from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.features.build_features import select_model_features
from src.features.build_features import create_reinjury_target
from src.features.build_features import add_history_features
from src.models.train import train_model
from src.models.train import train_random_forest

if __name__ == "__main__":
    df = load_raw_data("data/raw/player_injuries_impact.csv")
    df = preprocess_data(df)

    print("Shape after preprocess:", df.shape)
    print("\nAbsence days stats:")
    print(df["absence_days"].describe())


df = create_reinjury_target(df)


df = add_history_features(df)

print("\nNew features preview:")
print(df[["Name", "previous_injuries", "days_since_last_injury"]].head(10))

print("\nDistribution target:")
print(df["reinjury_90"].value_counts())


df = select_model_features(df)

print("\nFinal dataset shape:", df.shape)
print("\nFinal columns:")
print(df.columns)
model = train_model(df)
rf_model = train_random_forest(df)