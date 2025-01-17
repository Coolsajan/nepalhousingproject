import sys

from nepal_housing_project.exception import hosuingprojectException
from nepal_housing_project.logger import logging

import os
from nepal_housing_project.constants import DATABASE_NAME,MONGODB_URL
import pymongo

class MongoDBClient:
    '''
    Class Name : explort_data_into_feature_store
    Description : This menthod exports the dataframe from mongodb feature as dataframe

    Output : connection to mongo database
    On Falier : raises an exception
    
    '''
    client=None
    def __init___(self,database_name=DATABASE_NAME)->None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url=os.getenv(MONGODB_URL)
                if mongo_db_url is None:
                    raise Exception (f'Enviroment Key:{MONGODB_URL} is not set.')
                MongoDBClient.client=pymongo.MongoClient(mongo_db_url)
                self.client=MongoDBClient.client
                self.database=self.client[database_name]
                self.database_name=database_name
                logging.info("MongoDB connection establised sucessfully.")
        except Exception as e:
            raise hosuingprojectException(e,sys)
        
        