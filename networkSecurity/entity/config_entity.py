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
        
