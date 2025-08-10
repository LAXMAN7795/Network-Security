import os
import sys
import pandas as pd
import numpy as np

'''
This module contains constants related to the training pipeline of a network security system.
'''
TARGET_COLUMN:str='Result'
PIPELINE_NAME:str='networkSecurity'
ARTIFACT_DIRECTORY:str='Artifacts'
FILE_NAME:str='PhishingData.csv'

TRAIN_FILE_NAME:str='train.csv'
TEST_FILE_NAME:str='test.csv'

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")# Path to the schema file for data validation create data_schema directory and add schema.yaml file

'''
Data ingestion related constant starts with DATA_INGESTION_VAR_NAME
'''
DATA_INGESTION_COLLECTION_NAME:str='NetworkData'
DATA_INGESTION_DATABASE_NAME:str='LAXMANAI'
DATA_INGESTION_DIRECTORY_NAME:str='data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIRECTORY:str='feature_store'
DATA_INGESTION_INGESTED_DIRECTORY:str='ingested_data'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.2

'''
Data validation related constant starts with DATA_VALIDATION_VAR_NAME
'''
DATA_VALIDATION_DIR_NAME:str='data_validation'
DATA_VALIDATION_VALID_DIR:str='validated'
DATA_VALIDATION_INVALID_DIR:str='invalid'
DATA_VALIDATION_DRIFT_REPORT_DIR:str='drift_report'
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str='drift_report.yaml'

'''Data transformation related constant starts with DATA_TRANSFORMATION_VAR_NAME
'''
DATA_TRANSFORMATION_DIR_NAME:str='data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DIR:str='transformed'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str='transformed_object'
PREPROCESSING_OBJECT_FILE_NAME:str = 'preprocessing.pkl'

#KNN imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS:dict={
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform"
}