import yaml
from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
import os
import sys
import numpy as np
# import dill
# import pickle

def read_yaml_file(file_path: str) -> dict:# Reads a YAML file and returns its content as a dictionary.
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def write_yaml_file(file_path: str, content: object, replace:bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)