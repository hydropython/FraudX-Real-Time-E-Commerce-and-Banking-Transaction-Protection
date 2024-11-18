import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class eda_fruaddata:
    def __init__(self, file_path):
        """
        Initialize the class with the dataset.
        """
        self.data = pd.read_csv(file_path)
    
    def describe_data(self):
        """
        Provide basic statistics and data types.
        """
        print("Data Overview:")
        print(self.data.head(), "\n")
        
        print("Data Info:")
        print(self.data.info(), "\n")
        
        print("Statistical Summary:")
        print(self.data.describe(), "\n")
        
    def check_missing_values(self):
        """
        Check for missing values in the dataset.
        """
        missing = self.data.isnull().sum()
        print("Missing Values:")
        print(missing[missing > 0])
    
    def visualize_distributions(self):
        """
        Visualize the distributions of numerical columns.
        """
        num_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[num_cols].hist(bins=15, figsize=(15, 10), layout=(2, 3))
        plt.tight_layout()
        plt.show()
    
    def analyze_categorical_data(self):
        """
        Analyze the categorical columns with bar plots.
        """
        cat_cols = self.data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            plt.figure(figsize=(8, 5))
            sns.countplot(data=self.data, y=col, order=self.data[col].value_counts().index)
            plt.title(f"Distribution of {col}")
            plt.show()
    
    def correlation_matrix(self):
        """
        Compute and visualize the correlation matrix for numerical columns.
        """
        corr = self.data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        plt.show()
    
    def outlier_detection(self, column):
        """
        Detect and visualize outliers for a given numerical column using a boxplot.
        """
        if column in self.data.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=self.data, x=column)
            plt.title(f"Outlier Detection in {column}")
            plt.show()
        else:
            print(f"Column '{column}' not found in the dataset.")