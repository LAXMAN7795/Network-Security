import os
import sys

from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging

from networkSecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networkSecurity.entity.config_entity import DataTransformationConfig, ModelTrainerConfig

from networkSecurity.utils.ml_utils.model.estimator import NetworkModel
from networkSecurity.utils.main_utils.utils import save_object, load_object
from networkSecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networkSecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, model, classification_train_metric, X_sample):
        with mlflow.start_run():
            mlflow.log_metric("f1_score", classification_train_metric.f1_score)
            mlflow.log_metric("precision", classification_train_metric.precision)
            mlflow.log_metric("recall", classification_train_metric.recall)
            mlflow.sklearn.log_model(model, name="model", input_example=X_sample)

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=500),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
                "Random Forest": RandomForestClassifier(verbose=1, class_weight='balanced', max_depth=10),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier()
            }

            params = {
                "Logistic Regression": {},
                "KNN": {'n_neighbors': [3, 5, 7]},
                "Decision Tree": {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, None]},
                "Random Forest": {'n_estimators': [50, 100], 'max_depth': [5, 10]},
                "Gradient Boosting": {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
                "AdaBoost": {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
            }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # To get the best score
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            y_train_pred = best_model.predict(X_train)

            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            # Track the mlflow
            self.track_mlflow(best_model,classification_train_metric, X_sample=X_train[:1])


            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # Track the mlflow
            self.track_mlflow(best_model,classification_test_metric, X_sample=X_test[:1])

            preprocesser = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocesser=preprocesser, model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metrics_artifact=classification_train_metric,
                test_metrics_artifact=classification_test_metric
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_array = load_numpy_array_data(train_file_path)
            test_array = load_numpy_array_data(test_file_path)

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            train_model_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return train_model_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)