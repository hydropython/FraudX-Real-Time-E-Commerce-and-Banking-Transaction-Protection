import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from FraudDataProcessor import FraudDataProcessor  # Assuming the class is in this module

class FraudDataBalancer:
    def __init__(self, target=None, processor=None):
        """
        Initialize the FraudDataBalancer class with optional target column and processor.
        
        :param target: str, name of the target column (fraud indicator)
        :param processor: FraudDataProcessor, an instance of the data processor class
        """
        self.target = target
        self.processor = processor if processor else FraudDataProcessor(target)  # Default to FraudDataProcessor

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data using the provided processor (FraudDataProcessor).
        
        :param df: DataFrame to preprocess
        :return: Processed DataFrame
        """
        return self.processor.process_data(df)

    def random_undersampling(self, df):
        # Separate majority and minority classes
        fraud = df[df[self.target] == 1]
        non_fraud = df[df[self.target] == 0]

        # Ensure there's at least one fraud case
        if fraud.shape[0] == 0:
            raise ValueError("No fraud cases found in the data.")

        # Downsample the majority class (non-fraud)
        if non_fraud.shape[0] > fraud.shape[0]:
            non_fraud_downsampled = resample(non_fraud,
                                            replace=False,
                                            n_samples=fraud.shape[0],  # Match the minority class
                                            random_state=42)
        else:
            non_fraud_downsampled = non_fraud

        # Combine minority class and downsampled majority class
        balanced_data = pd.concat([fraud, non_fraud_downsampled])

        return balanced_data
        
    def combined_undersampling_oversampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform combined under-sampling and over-sampling (SMOTE) to balance the data.
        
        :param df: DataFrame to balance
        :return: Balanced DataFrame
        """
        # Preprocess the data first
        df = self.preprocess_data(df)
        
        # Separate fraud and non-fraud data
        fraud = df[df[self.target] == 1]
        non_fraud = df[df[self.target] == 0]
        
        # Exclude non-numeric columns before applying SMOTE
        non_fraud_numeric = non_fraud.drop(columns=[self.target, 'date'], errors='ignore')
        fraud_numeric = fraud.drop(columns=[self.target, 'date'], errors='ignore')

        # Perform SMOTE Oversampling on the numeric columns of the fraud class
        smote = SMOTE(random_state=42)
        fraud_oversampled, _ = smote.fit_resample(fraud_numeric, fraud[self.target])

        # Downsample the non-fraud class (undersampling) to match the oversampled fraud class
        non_fraud_downsampled = non_fraud.sample(n=fraud_oversampled.shape[0], random_state=42)

        # Combine the downsampled majority class and the oversampled minority class
        balanced_data = pd.concat([fraud_oversampled, non_fraud_downsampled])

        # Display the class distribution of the balanced dataset
        self.display_class_distribution(balanced_data)

        # Plot the distribution of the classes after balancing
        self.plot_class_distribution(balanced_data)

        return balanced_data

    def display_class_distribution(self, balanced_data: pd.DataFrame):
        """
        Display the distribution of the target variable ('class') after balancing.
        
        :param balanced_data: DataFrame to display class distribution
        """
        print("Class distribution after balancing:")
        print(balanced_data[self.target].value_counts())

    def plot_class_distribution(self, balanced_data: pd.DataFrame):
        """
        Plot the distribution of the target variable ('class') after balancing.
        
        :param balanced_data: DataFrame to plot class distribution
        """
        class_counts = balanced_data[self.target].value_counts()
        class_labels = class_counts.index
        class_sizes = class_counts.values
        
        # Create a bar plot for class distribution
        plt.figure(figsize=(6, 4))
        plt.bar(class_labels, class_sizes, color=['green', 'red'])
        plt.title('Class Distribution After Balancing')
        plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
        plt.ylabel('Number of Samples')
        plt.xticks(class_labels, ['Non-Fraud', 'Fraud'])
        plt.show()

    def save_to_csv(self, balanced_data: pd.DataFrame, filename="balanced_fruaddata.csv"):
        """
        Save the balanced dataset to a CSV file in the '../Data' directory.
        
        :param balanced_data: DataFrame to save
        :param filename: str, name of the file to save
        """
        save_path = f"../Data/{filename}"
        balanced_data.to_csv(save_path, index=False)
        print(f"Balanced dataset saved to {save_path}")