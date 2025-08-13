from networkSecurity.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import os
import sys
from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging

class NetworkModel:
    def __init__(self,preprocesser,model):
        try:
            self.preprocesser = preprocesser
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def predict(self, X):
        try:
            X_transformed = self.preprocesser.transform(X)
            predictions = self.model.predict(X_transformed)
            return predictions
        except Exception as e:
            logging.error(f"Error occurred while making predictions: {e}")
            raise NetworkSecurityException(e, sys)