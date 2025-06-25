import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd
from src.utils import save_model
import logging
from src.preprocess import DataPreprocessor, walk_forward_splits
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import platform
import shutil
import os
import glob
logger = logging.getLogger(__name__)
def get_xgb_params():
    return {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }
def get_xgb_device():
    try:
        if platform.system() == 'Darwin' and xgb.core._has_mps:
            return 'mps'
    except Exception:
        pass
    return 'cpu'
class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'svm': SVC(random_state=42, probability=True)
        }
        xgb_device = get_xgb_device()
        self.models['xgboost'] = xgb.XGBClassifier(
            tree_method='hist',
            device=xgb_device,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )    
    def train_xgboost_with_tuning(self, X_train, y_train, X_test, y_test):
        param_dist = get_xgb_params()
        xgb_device = get_xgb_device()
        base_model = xgb.XGBClassifier(
            tree_method='hist',
            device=xgb_device,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=10,
            scoring='accuracy',
            n_jobs=-1,
            cv=3,
            verbose=1,
            random_state=42
        )
        logger.info(f"Starting XGBoost hyperparameter tuning on device: {xgb_device}")
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        logger.info(f"Best XGBoost params: {search.best_params_}")
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        with mlflow.start_run(run_name="xgboost_tuned"):
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            })
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="xgboost_tuned",
                registered_model_name="stock-predictor-xgb"
            )
        model_path = save_model(best_model, "xgboost_tuned")
        logger.info(f"xgboost_tuned training completed - Accuracy: {accuracy:.4f}")
        return best_model, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
    def train_model(self, model_name: str, X_train, y_train, X_test, y_test):
        if model_name == 'xgboost':
            return self.train_xgboost_with_tuning(X_train, y_train, X_test, y_test)
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        with mlflow.start_run(run_name=f"{model_name}_training"):
            model = self.models[model_name]
            mlflow.log_params(model.get_params())
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            })
            mlflow.sklearn.log_model(model, model_name)
            model_path = save_model(model, model_name)
            logger.info(f"{model_name} training completed - Accuracy: {accuracy:.4f}")
            return model, {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }
    def train_all_models(self, X_train, y_train, X_test, y_test):
        results = {}
        mlflow.set_experiment("Stock_Prediction_Models")
        for model_name in self.models.keys():
            model, metrics = self.train_model(model_name, X_train, y_train, X_test, y_test)
            results[model_name] = {
                'model': model,
                'metrics': metrics
            }
        best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['accuracy'])
        best_model = results[best_model_name]['model']
        logger.info(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['metrics']['accuracy']:.4f}")
        return results, best_model_name
def walk_forward_validate_xgboost(data, n_splits=5):
    preprocessor = DataPreprocessor()
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'MACD_signal',
        'BB_upper', 'BB_lower', 'Volume_ratio',
        'Price_change', 'High_Low_pct', 'Open_Close_pct',
        'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5',
        'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_5',
        'Day_of_week', 'Month', 'Quarter', 'SP500_return', 'QQQ_return', 'DIA_return',
        'SP500_return_lag_1', 'SP500_return_lag_2', 'SP500_return_lag_3', 'SP500_return_lag_5',
        'Relative_strength', 'Symbol_encoded'
    ]
    data['Symbol_encoded'] = preprocessor.label_encoder.fit_transform(data['Symbol'])
    X = data[feature_columns]
    y = data['Target']
    X = X.sort_index()
    y = y.sort_index()
    results = []
    for i, (train_idx, test_idx) in enumerate(walk_forward_splits(X, n_splits=n_splits)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train_scaled = preprocessor.scaler.fit_transform(X_train)
        X_test_scaled = preprocessor.scaler.transform(X_test)
        xgb_device = get_xgb_device()
        model = xgb.XGBClassifier(
            tree_method='hist',
            device=xgb_device,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"Fold {i+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        results.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_prec = np.mean([r['precision'] for r in results])
    avg_rec = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    print(f"\nWalk-forward validation (XGBoost, {n_splits} folds):")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average Precision: {avg_prec:.4f}")
    print(f"Average Recall: {avg_rec:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
def main():
    preprocessor = DataPreprocessor()
    try:
        data = pd.read_csv("data/processed/processed_stock_data.csv")
    except FileNotFoundError:
        logger.error("Processed data file not found. Please run preprocessing first.")
        return
    if data.empty:
        logger.error("Processed data is empty. Cannot train model.")
        return
    X_train, X_test, y_train, y_test, features = preprocessor.prepare_training_data(data)
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Training or test data is empty. Cannot train model.")
        return
    trainer = ModelTrainer()
    results, best_model_name = trainer.train_all_models(X_train, y_train, X_test, y_test)
    os.makedirs("dvc_models", exist_ok=True)
    joblib.dump(preprocessor.scaler, "dvc_models/scaler.pkl")
    joblib.dump(preprocessor.label_encoder, "dvc_models/label_encoder.pkl")
    joblib.dump(features, "dvc_models/feature_columns.pkl")
    for old_file in glob.glob("models/*.pkl"):
        os.remove(old_file)    
    os.makedirs("models", exist_ok=True)
    for fname in os.listdir("dvc_models"):
        shutil.copy(os.path.join("dvc_models", fname), "models/")
    logger.info("Model training completed successfully")
    print("\n--- Walk-forward validation (XGBoost) ---")
    walk_forward_validate_xgboost(data, n_splits=10)
if __name__ == "__main__":
    main()
