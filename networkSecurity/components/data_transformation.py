import sys
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networkSecurity.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS

from networkSecurity.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact

from networkSecurity.entity.config_entity import DataValidationConfig, DataTransformationConfig
from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
from networkSecurity.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            logging.info("Data read successfully")
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformation_object(cls) -> Pipeline:
        '''
        It initialises a KNN Imputer object with the parameter specified in the training_pipeline.py file
        and returns a Pipeline object with the imputer.

        args:
            cls: DataTransformation
        returns:
            Pipeline: A scikit-learn Pipeline object with the KNN imputer.
        '''
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info("KNN Imputer created with parameters: {}".format(DATA_TRANSFORMATION_IMPUTER_PARAMS))
            processer: Pipeline = Pipeline([("imputer", imputer)])
            return processer
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Data Transformation initiated")
        try:
            logging.info("Starting data transformation")
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_train_df.replace(-1,0)

            # Testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)

            preprocessor = self.get_data_transformation_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_feature_train_df = preprocessor_object.transform(input_feature_train_df)
            transformed_input_feature_test_df = preprocessor_object.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_feature_test_df, np.array(target_feature_test_df)]

            # save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            save_object("final_model/preprocessor.pkl", preprocessor_object)

            # preparing artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info("Data Transformation completed successfully")
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)