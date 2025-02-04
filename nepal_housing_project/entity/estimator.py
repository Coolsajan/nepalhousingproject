import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from nepal_housing_project.exception import hosuingprojectException
from nepal_housing_project.logger import logging

class CitytValueMapping:
    def __init__(self):
        self.kathmandu:int = 0
        self.bhaktapur:int = 1
        self.lalitpur:int = -1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))
    

class HousingModel:
    def __init__(self,preprocessing_object: Pipeline,trained_model_object:object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object=preprocessing_object
        self.trained_model_object=trained_model_object

    def predict(self,dataframe:DataFrame)->DataFrame:
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        try:
            logging.info("Using the trained model to get prediction.")

            transformed_feature=self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions.")
            return self.trained_model_object.predict(transformed_feature)
        
        except Exception as e:
            raise hosuingprojectException(e,sys) from e
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"

