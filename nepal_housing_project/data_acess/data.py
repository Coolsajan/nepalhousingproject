from nepal_housing_project.configuration.mongobd_connection import MongoDBClient
from nepal_housing_project.constants import DATABASE_NAME
from nepal_housing_project.logger import logging
from nepal_housing_project.exception import hosuingprojectException

import pandas as pd
import numpy as np
import sys
from typing import Optional



class housingdata:
    '''
    Extract housing data from mongo database and record as pandas dataframe.
    '''
    def __init__(self):
        try:
            self.mongo_client=MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise hosuingprojectException(e,sys)
        

    def export_collection_as_Dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        try:
            '''
            export entire collection into dataframe
            return pd.dateframe
                        
            '''
            if database_name is None:
                collection=self.mongo_client.client[collection_name]
            else:
                collection=self.mongo_client.client[database_name][collection_name]
            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(column=['_id'],axis=1)
            df.replace({'na':np.nan},inplace=True)
            return df
        except Exception as e:
            raise hosuingprojectException(e,sys)