from networkSecurity.entity.artifact_entity import ClassificationMetricArtifact
from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
import sys
from sklearn.metrics import f1_score, precision_score, recall_score

def get_classification_score(y_true,y_pred) -> ClassificationMetricArtifact:
    try:
        f1 = f1_score(y_true, y_pred,average='weighted')
        precision = precision_score(y_true, y_pred,average='weighted')
        recall = recall_score(y_true, y_pred,average='weighted')

        classification_metric_artifact = ClassificationMetricArtifact(
            f1_score=f1,
            precision=precision,
            recall=recall
        )
        return classification_metric_artifact
    except Exception as e:
        logging.error(f"Error occurred while calculating classification metrics: {e}")
        raise NetworkSecurityException(e, sys)