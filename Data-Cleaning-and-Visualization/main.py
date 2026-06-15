import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())

df.drop_duplicates(inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

sns.histplot(df["Age"], bins=20)
plt.title("Age Distribution")
plt.show()