from networkSecurity.components.data_ingestion import DataIngestion

from networkSecurity.logging.logger import logging
from networkSecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from networkSecurity.exception.exception import NetworkSecurityException

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Starting data ingestion process.")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(f"Data ingestion artifact: {data_ingestion_artifact}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")