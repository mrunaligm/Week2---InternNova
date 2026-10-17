import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Task 1: NumPy
one_d = np.array([10, 20, 30, 40, 50])
two_d = np.array([[1, 2, 3], [4, 5, 6]])
print("1D:", one_d)
print("2D:\n", two_d)
print("Index:", one_d[0], "Slice:", one_d[1:4])
print("Shape:", two_d.shape, "Dimensions:", two_d.ndim)
print("Mean/max/min/sum:", one_d.mean(), one_d.max(), one_d.min(), one_d.sum())

# Tasks 2-3: Pandas and cleaning
df = pd.read_csv("student_performance_raw.csv")
print(df.head(), df.tail(), df.shape, df.dtypes)
print(df.describe(include="all"))
for col in ["Maths", "Science", "English", "Attendance"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("Missing before:\n", df.isna().sum())
print("Duplicates before:", df.duplicated().sum())
df = df.drop_duplicates().copy()
for col in ["Maths", "Science", "English"]:
    df[col] = df[col].fillna(df[col].mean())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].median())
subjects = ["Maths", "Science", "English"]
df["Average_Marks"] = df[subjects].mean(axis=1)
df["Result"] = np.where(df[subjects].min(axis=1) >= 35, "Pass", "Fail")
print("Missing after:\n", df.isna().sum())
print("Average:", df[subjects].to_numpy().mean())
print("Highest:", df[subjects].max().max(), "Lowest:", df[subjects].min().min())
print("Pass:", (df.Result == "Pass").sum(), "Fail:", (df.Result == "Fail").sum())

# Task 4: Four visualizations
plt.figure(); plt.bar(df.Name.head(10), df.Average_Marks.head(10)); plt.title("Average Marks of First 10 Students"); plt.xticks(rotation=35); plt.tight_layout(); plt.show()
plt.figure(); plt.plot(df.Student_ID, df.Average_Marks, marker="o"); plt.title("Average Marks by Student ID"); plt.xticks(rotation=45); plt.tight_layout(); plt.show()
plt.figure(); plt.hist(df.Average_Marks, bins=6, edgecolor="black"); plt.title("Distribution of Average Marks"); plt.tight_layout(); plt.show()
plt.figure(); plt.scatter(df.Attendance, df.Average_Marks); plt.title("Attendance vs Average Marks"); plt.xlabel("Attendance"); plt.ylabel("Average marks"); plt.tight_layout(); plt.show()
