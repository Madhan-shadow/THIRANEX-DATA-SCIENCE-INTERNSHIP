import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("retail_sales_dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\nSummary:")
print(df.describe())

df.groupby("Product Category")["Total Amount"].sum().sort_values().plot(kind="bar")

plt.title("Sales by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.tight_layout()

plt.savefig("sales_by_category.png")
plt.show()

print("\nRetail Analysis Completed Successfully!")