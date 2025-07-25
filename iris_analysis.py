# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Optional: Set seaborn style
sns.set(style="whitegrid")


# Task 1: Load and Explore the Dataset

try:
    # Load Iris dataset from sklearn
    iris_data = load_iris()
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)
    
    # Display the first few rows
    print("🔍 First 5 rows of dataset:")
    print(df.head())
    
    # Data types and missing values
    print("\n📊 Data Types:")
    print(df.dtypes)
    
    print("\n❓ Missing values:")
    print(df.isnull().sum())
    
    # No missing values in this dataset, but we’ll show how to handle them
    df = df.dropna()  # just in case

except FileNotFoundError:
    print("❌ Dataset file not found.")
except Exception as e:
    print("⚠️ An error occurred:", str(e))


#task 2
print("\n📈 Basic Statistics:")
print(df.describe())

# Group by species and calculate mean of numerical columns
print("\n📊 Mean values grouped by species:")
print(df.groupby('species').mean())

# Observations
print("\n🧠 Observations:")
print("Setosa has smallest petal sizes, Virginica the largest. Sepal lengths and widths vary more subtly.")


# Task 3: Data Visualization

# Line Chart (Simulate trend by plotting sepal length across index)
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title("Line Chart: Sepal Length over Entries")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# Bar Chart: Average petal length per species
plt.figure(figsize=(8, 4))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(8, 4))
plt.hist(df['sepal width (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()
