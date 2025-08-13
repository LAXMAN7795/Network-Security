from datetime import datetime
import os
from networkSecurity.constant import training_pipeline

print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIRECTORY)

class TrainingPipelineConfig:
    def __init__(self):
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_directory: str = training_pipeline.ARTIFACT_DIRECTORY
        self.timestamp: str = datetime.now().strftime("%Y%m%d%H%M%S")
        self.artifact_directory_path: str = os.path.join(self.artifact_directory, self.timestamp)

class DataIngestionConfig:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_directory: str = os.path.join(
            training_pipeline_config.artifact_directory_path,
            training_pipeline.DATA_INGESTION_DIRECTORY_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_directory,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIRECTORY,
            training_pipeline.FILE_NAME
        )
        self.train_file_path: str = os.path.join(
            self.data_ingestion_directory,
            training_pipeline.DATA_INGESTION_INGESTED_DIRECTORY,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.test_file_path: str = os.path.join(
            self.data_ingestion_directory,
            training_pipeline.DATA_INGESTION_INGESTED_DIRECTORY,
            training_pipeline.TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        
class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_directory: str = os.path.join(
            training_pipeline_config.artifact_directory_path,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_directory: str = os.path.join(
            self.data_validation_directory,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_directory: str = os.path.join(
            self.data_validation_directory,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_directory,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_directory,
            training_pipeline.TEST_FILE_NAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_directory,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_directory,
            training_pipeline.TEST_FILE_NAME
        )
        self.drift_report_directory: str = os.path.join(
            self.data_validation_directory,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR
        )
        self.drift_report_file_path: str = os.path.join(
            self.drift_report_directory,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_directory: str = os.path.join(
            training_pipeline_config.artifact_directory_path,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_directory,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_directory,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace('csv', 'npy')
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_directory,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR,
            training_pipeline.TEST_FILE_NAME.replace('csv', 'npy')
        )

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_directory: str = os.path.join(
            training_pipeline_config.artifact_directory_path,
            training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_directory: str = os.path.join(
            self.model_trainer_directory,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR
        )
        self.trained_model_file_path: str = os.path.join(
            self.trained_model_directory,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME
        )
        self.expected_score: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold: float = training_pipeline.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD