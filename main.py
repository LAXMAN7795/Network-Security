from networkSecurity.components.data_ingestion import DataIngestion
import sys
from networkSecurity.components.data_validation import DataValidation
from networkSecurity.logging.logger import logging
from networkSecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from networkSecurity.exception.exception import NetworkSecurityException

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Starting data ingestion process.")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(f"Data ingestion artifact: {data_ingestion_artifact}")
        logging.info("Data ingestion completed successfully.")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Starting data validation process.")
        data_validation_artifact = data_validation.initiate_data_validation()
        print(f"Data validation artifact: {data_validation_artifact}")
        logging.info("Data validation completed successfully.")
    except Exception as e:
        raise NetworkSecurityException(e, sys)