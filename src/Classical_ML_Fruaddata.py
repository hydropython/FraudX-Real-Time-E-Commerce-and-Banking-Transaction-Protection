import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import os

class FraudModelingPipeline:
    def __init__(self, df, target_column, dataset_name="fraud-data"):
        self.df = self.remove_problematic_columns(df)
        self.target_column = target_column
        self.dataset_name = dataset_name
        self.X = pd.get_dummies(self.df.drop(columns=[target_column]), drop_first=True)
        self.y = self.df[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def remove_problematic_columns(self, df):
        """Remove columns that cause issues (e.g., non-numeric types, too many NaNs)."""
        # Example criteria: drop columns with non-numeric data types or too many missing values
        numeric_df = df.select_dtypes(include=[np.number])
        # Optionally, set a threshold for NaNs
        threshold = 0.5 * len(df)  # for example, drop columns with more than 50% NaNs
        cleaned_df = numeric_df.loc[:, numeric_df.isnull().sum() < threshold]
        return cleaned_df

    def train_models(self):
        """Train various models and log their performance using MLflow."""
        mlflow.set_experiment(f"{self.dataset_name}_model_training")
        
        active_run = mlflow.active_run()
        if active_run is not None:
            print(f"Ending active run: {active_run.info.run_id}")
            mlflow.end_run()

        with mlflow.start_run():
            # Logistic Regression
            self.logistic_model = LogisticRegression(max_iter=1000)
            self.logistic_model.fit(self.X_train, self.y_train)
            self.log_metrics(self.y_test, self.logistic_model.predict(self.X_test), "Logistic Regression")
            mlflow.sklearn.log_model(self.logistic_model, artifact_path="logistic_regression_model")

            # Decision Tree
            self.decision_tree_model = DecisionTreeClassifier(random_state=42)
            self.decision_tree_model.fit(self.X_train, self.y_train)
            self.log_metrics(self.y_test, self.decision_tree_model.predict(self.X_test), "Decision Tree")
            mlflow.sklearn.log_model(self.decision_tree_model, artifact_path="decision_tree_model")

            # Random Forest
            self.random_forest_model = RandomForestClassifier(random_state=42)
            self.random_forest_model.fit(self.X_train, self.y_train)
            self.log_metrics(self.y_test, self.random_forest_model.predict(self.X_test), "Random Forest")
            mlflow.sklearn.log_model(self.random_forest_model, artifact_path="random_forest_model")

            # Gradient Boosting
            self.gradient_boosting_model = GradientBoostingClassifier(random_state=42)
            self.gradient_boosting_model.fit(self.X_train, self.y_train)
            self.log_metrics(self.y_test, self.gradient_boosting_model.predict(self.X_test), "Gradient Boosting")
            mlflow.sklearn.log_model(self.gradient_boosting_model, artifact_path="gradient_boosting_model")

    def log_metrics(self, y_true, y_pred, model_name):
        """Log metrics for a given model in MLflow."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"--- {model_name} ---")
        print(classification_report(y_true, y_pred))

        # Logging metrics and parameters to MLflow in a nested run
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Optionally log confusion matrix as a PNG image
            confusion_matrix_img = self.plot_confusion_matrix(y_true, y_pred, model_name)
            mlflow.log_artifact(confusion_matrix_img)
            os.remove(confusion_matrix_img)  # Clean up the local image file
            
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Generate and save confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, cmap='Blues')
        plt.title(f"Confusion Matrix for {model_name}")
        plt.colorbar()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.xticks(np.arange(2), ['Not Fraud', 'Fraud'])
        plt.yticks(np.arange(2), ['Not Fraud', 'Fraud'])
        
        img_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(img_path)
        plt.close()
        return img_path

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning using RandomizedSearchCV for Random Forest and Gradient Boosting"""
        # Random Forest Hyperparameter Tuning
        rf_param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        rf_random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=rf_param_distributions,
            n_iter=50,
            scoring='accuracy',
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        rf_random_search.fit(self.X_train, self.y_train)
        print(f"Best Random Forest Parameters: {rf_random_search.best_params_}")
        self.rf_best_model = rf_random_search.best_estimator_
        rf_predictions = self.rf_best_model.predict(self.X_test)
        
        # Evaluation and logging to MLflow
        self.log_metrics(self.y_test, rf_predictions, "Random Forest (Tuned)")
        mlflow.sklearn.log_model(self.rf_best_model, artifact_path="random_forest_best_model")
        
        # Gradient Boosting Hyperparameter Tuning
        gb_param_distributions = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.05],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.6, 0.8, 1.0]
        }
        
        gb_random_search = RandomizedSearchCV(
            estimator=GradientBoostingClassifier(random_state=42),
            param_distributions=gb_param_distributions,
            n_iter=50,
            scoring='accuracy',
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        gb_random_search.fit(self.X_train, self.y_train)
        print(f"Best Gradient Boosting Parameters: {gb_random_search.best_params_}")
        self.gb_best_model = gb_random_search.best_estimator_
        gb_predictions = self.gb_best_model.predict(self.X_test)
        
        # Evaluation and logging to MLflow
        self.log_metrics(self.y_test, gb_predictions, "Gradient Boosting (Tuned)")
        mlflow.sklearn.log_model(self.gb_best_model, artifact_path="gradient_boosting_best_model")

    def shap_analysis(self):
        """Perform SHAP analysis to explain model predictions."""
        explainer = shap.Explainer(self.gb_best_model, self.X_train)
        shap_values = explainer(self.X_test)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, self.X_test, feature_names=self.X.columns, show=False)
        shap_img_path = 'shap_summary_plot.png'
        plt.savefig(shap_img_path, dpi=300, bbox_inches='tight')
        plt.close()

        mlflow.log_artifact(shap_img_path)
        os.remove(shap_img_path)

    def lime_analysis(self):
        """Perform LIME analysis to explain model predictions."""
        lime_explainer = LimeTabularExplainer(
            training_data=np.array(self.X_train),
            feature_names=self.X.columns,
            class_names=['Not Fraud', 'Fraud'],
            mode='classification'
        )
        i = 0
        exp = lime_explainer.explain_instance(data_row=self.X_test.iloc[i], predict_fn=self.gb_best_model.predict_proba)
        lime_img_path = f'lime_explanation_instance_{i}.png'
        exp.save_to_file(lime_img_path)
        
        mlflow.log_artifact(lime_img_path)
        os.remove(lime_img_path)
    def shap_force_plot(self, instance_index=0):
        """
        Generates a SHAP force plot for a specific instance and displays it using Matplotlib.
        
        Args:
        - instance_index (int): The index of the instance to visualize (default is 0).
        """
        # Check if the gradient boosting model has been trained
        if not hasattr(self, 'gradient_boosting_model'):
            raise AttributeError("The model is not trained. Please train the model by running `train_models()` first.")
        
        # Initialize the SHAP explainer for the gradient boosting model
        explainer = shap.TreeExplainer(self.gradient_boosting_model)

        # Calculate SHAP values for the test set
        shap_values = explainer.shap_values(self.X_test)

        # Determine SHAP values and expected value based on binary or multiclass model
        if isinstance(shap_values, list) and len(shap_values) > 1:  # Binary classification
            shap_values_class = shap_values[1]  # SHAP values for the positive class
            expected_value = explainer.expected_value[1]
        else:  # Multiclass or regression
            shap_values_class = shap_values
            expected_value = explainer.expected_value

        # Generate the SHAP force plot for the specific instance
        shap.initjs()  # Initialize JavaScript for interactive SHAP plots
        force_plot = shap.force_plot(expected_value, shap_values_class[instance_index], self.X_test.iloc[instance_index], matplotlib=True)

        # Display the force plot as a Matplotlib figure
        plt.show()