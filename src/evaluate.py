import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
import joblib
from utils import load_model
import logging
from preprocess import DataPreprocessor
logger = logging.getLogger(__name__)
class ModelEvaluator:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.scaler = joblib.load("models/scaler.pkl")
        self.label_encoder = joblib.load("models/label_encoder.pkl")
        self.feature_columns = joblib.load("models/feature_columns.pkl")
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('models/confusion_matrix.png')
        plt.show()        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('models/roc_curve.png')
        plt.show()
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'auc': roc_auc
        }
    def feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance.head(15), x='importance', y='feature')
            plt.title('Top 15 Feature Importances')
            plt.tight_layout()
            plt.savefig('models/feature_importance.png')
            plt.show()
            if 'SP500_return' in importance.head(15)['feature'].values:
                logger.info("SP500_return is among the top 15 most important features!")
            return importance
        else:
            logger.info("Model doesn't support feature importance")
            return None
def main():
    preprocessor = DataPreprocessor()
    try:
        data = pd.read_csv("data/processed/processed_stock_data.csv")
    except FileNotFoundError:
        logger.error("Processed data file not found. Please run preprocessing and training first.")
        return
    if data.empty:
        logger.error("Processed data is empty. Cannot evaluate model.")
        return
    X_train, X_test, y_train, y_test, features = preprocessor.prepare_training_data(data)
    try:
        evaluator = ModelEvaluator("models/random_forest_latest.pkl") 
    except Exception as e:
        logger.error(f"Error loading model for evaluation: {e}")
        return
    results = evaluator.evaluate_model(X_test, y_test)
    importance = evaluator.feature_importance()
    logger.info("Model evaluation completed successfully")
if __name__ == "__main__":
    main() 
