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
    

