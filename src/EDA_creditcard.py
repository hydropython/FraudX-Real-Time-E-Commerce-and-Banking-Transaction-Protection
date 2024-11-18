import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px

class EDAcreditcard:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDAcreditcard class with a DataFrame.
        """
        self.data = data

    def summarize_data(self):
        """
        Display basic statistics and information about the dataset.
        """
        print("Dataset Overview:")
        print(self.data.info())
        print("\nSummary Statistics:")
        print(self.data.describe())

    def check_missing_values(self):
        """
        Check for missing values in the dataset.
        """
        missing = self.data.isnull().sum()
        print("\nMissing Values per Column:")
        print(missing[missing > 0])

    def plot_distributions(self, columns=None):
        """
        Plot the distributions of specified columns. If None, plot for all numerical columns.
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        for column in columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[column], kde=True, bins=30)
            plt.title(f'Distribution of {column}')
            plt.show()

    def correlation_heatmap(self):
        """
        Generate a heatmap to visualize the correlation between numerical features.
        """
        plt.figure(figsize=(12, 8))
        corr = self.data.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def perform_pca(self, n_components=2):
        """
        Perform PCA on the numerical columns of the dataset and output numerical results.
        """
        # Select numerical features and drop the 'Class' column
        numerical_data = self.data.select_dtypes(include=[np.number]).drop(['Class'], axis=1)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numerical_data)
        
        # Explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained Variance by each Principal Component: {explained_variance}")
        print(f"Total Explained Variance by {n_components} Components: {sum(explained_variance)}")
        
        # Create a DataFrame for PCA results
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
        pca_df['Class'] = self.data['Class']
        
        # If interactive plotting works
        try:
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Class',
                            title='PCA Scatter Plot',
                            color_discrete_sequence=px.colors.qualitative.Dark24)
            fig.show()
        except Exception as e:
            print(f"Interactive plotting failed: {e}")
            print("Displaying PCA scatter as a static plot.")
            plt.figure(figsize=(8, 6))
            for cls in pca_df['Class'].unique():
                subset = pca_df[pca_df['Class'] == cls]
                plt.scatter(subset['PC1'], subset['PC2'], label=f'Class {cls}', alpha=0.6)
            plt.title("PCA Scatter Plot")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend()
            plt.show()

    def plot_time_series(self):
        """
        Plot the time series of the 'Amount' column grouped by 'Class'.
        """
        if 'Time' in self.data.columns and 'Amount' in self.data.columns:
            self.data['Time'] = pd.to_datetime(self.data['Time'], errors='coerce')
            grouped = self.data.groupby('Class')
            for name, group in grouped:
                plt.figure(figsize=(10, 5))
                plt.plot(group['Time'], group['Amount'], label=f'Class {name}')
                plt.title('Time Series of Amount')
                plt.xlabel('Time')
                plt.ylabel('Amount')
                plt.legend()
                plt.show()
        else:
            print("'Time' or 'Amount' column is missing for time series analysis.")