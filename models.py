import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


class ModelTrainer:
    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=1.0),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "XGBoost": xgb.XGBRegressor(random_state=42),
            "LightGBM": lgb.LGBMRegressor(random_state=42),
        }
        self.results = {}

    def train_and_evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test):
        for name, model in self.models.items():
            print(f"\n{name} 학습 중...")

            # 모델 학습
            model.fit(X_train, y_train)

            # 예측
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)

            # 성능 평가
            train_mse = mean_squared_error(y_train, y_pred_train)
            val_mse = mean_squared_error(y_val, y_pred_val)
            test_mse = mean_squared_error(y_test, y_pred_test)

            train_r2 = r2_score(y_train, y_pred_train)
            val_r2 = r2_score(y_val, y_pred_val)
            test_r2 = r2_score(y_test, y_pred_test)

            self.results[name] = {
                "train_mse": train_mse,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "train_r2": train_r2,
                "val_r2": val_r2,
                "test_r2": test_r2,
            }

            print(f"{name} 결과:")
            print(f"Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
            print(f"Validation MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
            print(f"Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

            # 변수 중요도 시각화 (가능한 모델에 대해서만)
            if hasattr(model, "feature_importances_"):
                self.plot_feature_importance(model, X_train.columns, name)
            elif hasattr(model, "coef_"):
                self.plot_coefficients(model, X_train.columns, name)

    def plot_feature_importance(self, model, feature_names, model_name):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"{model_name} - Feature Importance")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_importance.png')
        plt.close()

    def plot_coefficients(self, model, feature_names, model_name):
        coef = model.coef_
        indices = np.argsort(np.abs(coef))[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"{model_name} - Coefficients")
        plt.bar(range(len(coef)), coef[indices])
        plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_coefficients.png')
        plt.close()

    def plot_results_comparison(self):
        # MSE 비교
        plt.figure(figsize=(12, 6))
        models = list(self.results.keys())
        test_mse = [self.results[m]["test_mse"] for m in models]

        plt.bar(models, test_mse)
        plt.title("Test MSE Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("model_comparison_mse.png")
        plt.close()

        # R² 비교
        plt.figure(figsize=(12, 6))
        test_r2 = [self.results[m]["test_r2"] for m in models]

        plt.bar(models, test_r2)
        plt.title("Test R² Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("model_comparison_r2.png")
        plt.close()
