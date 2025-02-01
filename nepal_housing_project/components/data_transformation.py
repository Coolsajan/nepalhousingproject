import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer,StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from nepal_housing_project.constants import TARGET_COLUMN,SCHEMA_FILE_PATH
from nepal_housing_project.entity.config_entity import DataTransformationConfig
from nepal_housing_project.entity.artifact_entity import DataingestionArtifact,DataValidationArtifact,DataTransformationArtifact

from nepal_housing_project.utils.main_utils import save_object,save_numpy_array_data,read_yaml_file

from nepal_housing_project.logger import logging
from nepal_housing_project.exception import hosuingprojectException
from nepal_housing_project.entity.estimator import CitytValueMapping

class DataTransformation:
    def __init__(self,data_ingestion_artifact:DataingestionArtifact,
                 data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationArtifact):
        """
        :prams data_ingestion_artifact : Output refrence of datainestion artifact stage
        :param data_transformation_config : configuration for data transformation
        """
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
            self._schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise hosuingprojectException(e,sys) 
        
    @staticmethod
    def read_data(file_path) ->pd.DataFrame:
        try: 
            return pd.read_csv(file_path)
        except Exception as e:
            raise hosuingprojectException(e,sys)
        
    def get_data_transformer_object(self)-> Pipeline:
    
        """
        Method Name : get_data_transformer_object
        Description : This method creates and return a data transformer object for the data

        Output : data transformer object is created and returned
        On failuer : write an exception log and raise exception
        """

        logging.info("Entred gate_data_transformer_object method of DataTransformation class")

        try:
            logging.info("Got numerical cols from schema config")

            oe_transformer=OrdinalEncoder()
            ohe_transformer=OneHotEncoder(sparse_output=False,handle_unknown="infrequent_if_exist")
            si_num_transformer=SimpleImputer(strategy="median")
            si_cate_transformer=SimpleImputer(strategy="most_frequent")


            logging.info("Initilization Ordinal Encoding,OneHot Encoding")

            oe_columns=self._schema_config['oe_columns']

            ohe_columns=self._schema_config['ohe_columns']
            si_num_columns=self._schema_config['si_num_columns']
            si_cate_columns=self._schema_config['si_cate_columns']
            powertransform_columns=self._schema_config['power_transformer']

            logging.info("Initilize Power Transformer")

            transfrom_pipe=Pipeline(steps=[
                ("transformer_column",PowerTransformer(method="yeo-johnson"))
            ])


            preprocessor=ColumnTransformer([
                ("si_cate",si_cate_transformer,si_cate_columns),
                ("si_num",si_num_transformer,si_num_columns),
                ("ordinal_encoder",oe_transformer,oe_columns),
                ("onehot_encoder",ohe_transformer,ohe_columns),
                ("powertransformer",transfrom_pipe,powertransform_columns)

                ]                
                ,remainder='passthrough')
            
            logging.info("Data Transformation .preprocess object achived.")
            return preprocessor
        except Exception as e:
            raise hosuingprojectException(e,sys)




    def initiate_data_transformation(self, )->DataTransformationArtifact:
        """
        Method Name : initate_data_transformation
        Description : This method iniatiate the data transformation component for the pipeline

        Output : data transformation is perfomed and preprocessor object is created
        On Failure : Writ an exception log and rasie an exception 
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data Transformation")
                preprocessor=self.get_data_transformer_object()
                logging.info("Got the processor object")

                train_df=DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df=DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
                train_df['LAND AREA']=train_df['LAND AREA'].replace(to_replace={0:np.nan,2.5:np.nan,4.2:np.nan,4.5:np.nan,9.25:np.nan})
                test_df['LAND AREA']=test_df['LAND AREA'].replace(to_replace={0:np.nan,2.5:np.nan,4.2:np.nan,4.5:np.nan,9.25:np.nan})



                train_df=train_df.dropna(subset=["PRICE","LAND AREA"])
                test_df=test_df.dropna(subset=["PRICE","LAND AREA"])



                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("GOT train feature and test feature of training dataset")


                input_feature_train_df["CITY"]=input_feature_train_df["CITY"].replace(CitytValueMapping()._asdict())

                target_feature_train_df=np.log(target_feature_train_df+1)

                logging.info("taget_feature_train log transformed")

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                input_feature_test_df['LAND AREA']=input_feature_test_df['LAND AREA'].replace(to_replace={0:np.nan,2.5:np.nan,4.2:np.nan,4.5:np.nan,9.25:np.nan})

                logging.info("City added on test dataset")


                input_feature_test_df["CITY"]=input_feature_test_df["CITY"].replace(CitytValueMapping()._asdict())

                target_feature_test_df=np.log(target_feature_test_df+1)

                logging.info("taget_feature_test log transformed")

                logging.info("Got tain feature and test feature of testing dataset")

                logging.info("Applying preprocessory object on traning dataframe and testing dataframe")
                print(input_feature_train_df)
                preprocessor.fit(input_feature_train_df)
                input_feature_train_arr=preprocessor.transform(input_feature_train_df)

                logging.info("Used the preprocessor object to fit transfrom train data")

                input_feature_test_arr=preprocessor.transform(input_feature_test_df)

                logging.info("Used the preprocessor object to fit transfrom test data")

                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)
                ]

                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


                save_object(self.data_transformation_config.transformed_object_file_path,preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)      



        except Exception as e:
            raise hosuingprojectException(e,sys) from e