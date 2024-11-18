import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class FraudDataProcessor:
    def __init__(self, target=None):
        """
        Initialize the FraudDataProcessor class with optional target column.
        
        :param target: str, name of the target column (fraud indicator)
        """
        self.target = target

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data including handling missing values, encoding categorical columns, and scaling numeric columns.
        
        :param df: DataFrame to preprocess
        :return: Preprocessed DataFrame
        """
        # Handle missing values (simple imputation with the mean for numeric columns)
        df = self.handle_missing_values(df)

        # Encode categorical variables (e.g., label encoding for the target column)
        df = self.encode_categorical_columns(df)

        # Scale numeric columns to ensure features have the same scale
        df = self.scale_numeric_columns(df)

        # Optionally: Drop non-informative or unnecessary columns (e.g., 'date', etc.)
        df = self.drop_unnecessary_columns(df)

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values by imputing with the mean for numeric columns.
        
        :param df: DataFrame to handle missing values
        :return: DataFrame with imputed values
        """
        imputer = SimpleImputer(strategy='mean')  # Can change to 'median' or other strategies
        df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))
        return df

    def encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables (e.g., target column) using Label Encoding.
        
        :param df: DataFrame to encode
        :return: DataFrame with encoded columns
        """
        if self.target and self.target in df.columns:
            label_encoder = LabelEncoder()
            df[self.target] = label_encoder.fit_transform(df[self.target])
        
        # Optionally encode other categorical features
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != self.target:
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
        
        return df

    def scale_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric columns to a standard range (e.g., using StandardScaler).
        
        :param df: DataFrame to scale
        :return: DataFrame with scaled numeric columns
        """
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        return df

    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop non-informative or unnecessary columns, like 'date' or 'ID' columns.
        
        :param df: DataFrame to drop columns from
        :return: DataFrame with dropped columns
        """
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        return df