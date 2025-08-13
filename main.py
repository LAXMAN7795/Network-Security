from networkSecurity.components.data_ingestion import DataIngestion
import sys
from networkSecurity.components.data_validation import DataValidation
from networkSecurity.components.data_transformation import DataTransformation
from networkSecurity.components.model_trainer import ModelTrainer
from networkSecurity.logging.logger import logging
from networkSecurity.entity.config_entity import ModelTrainerConfig, TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from networkSecurity.entity.config_entity import DataTransformationConfig
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
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Starting data transformation process.")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(f"Data transformation artifact: {data_transformation_artifact}")
        logging.info("Data transformation completed successfully.")

        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        logging.info("Starting model training process.")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        print(f"Model trainer artifact: {model_trainer_artifact}")
        logging.info("Model training completed successfully.")

    except Exception as e:
        raise NetworkSecurityException(e, sys)