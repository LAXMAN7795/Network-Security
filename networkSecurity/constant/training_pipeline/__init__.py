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

'''
Data ingestion related constant starts with DATA_INGESTION_VAR_NAME
'''
DATA_INGESTION_COLLECTION_NAME:str='NetworkData'
DATA_INGESTION_DATABASE_NAME:str='LAXMANAI'
DATA_INGESTION_DIRECTORY_NAME:str='data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIRECTORY:str='feature_store'
DATA_INGESTION_INGESTED_DIRECTORY:str='ingested_data'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.2

