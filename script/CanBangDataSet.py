import pandas as pd

df = pd.read_csv("health_classification_dataset.csv")

df_0 = df[df["is_health"] == 0]
df_1 = df[df["is_health"] == 1]

print("Trước:")
print(df["is_health"].value_counts())

# undersample class 0
df_0_down = df_0.sample(
    n=len(df_1),
    random_state=42
)

df_balanced = pd.concat([df_0_down, df_1])
df_balanced = df_balanced.sample(frac=1, random_state=42)

print("\nSau:")
print(df_balanced["is_health"].value_counts())

df_balanced.to_csv(
    "health_classification_balanced.csv",
    index=False,
    encoding="utf-8"
)
