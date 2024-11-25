import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feast import Field, FeatureStore, Entity, FeatureView, FileSource
from feast.types import Int64, Float64, String
from sklearn.preprocessing import OneHotEncoder
from feast.value_type import ValueType
from datetime import datetime, timedelta
import warnings
from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
warnings.filterwarnings('ignore')

@dataclass
class FeatureSchema:
    def __init__(self):
        self.schema_config = {
            'numerical_features': {
                'tenure': Int64,
                'MonthlyCharges': Float64,
                'TotalCharges': Float64
            },
            'categorical_features': {
                'gender': String,
                'SeniorCitizen': Int64,
                'Partner': String,
                'Dependents': String,
                'PhoneService': String,
                'MultipleLines': String,
                'InternetService': String,
                'OnlineSecurity': String,
                'DeviceProtection': String,
                'TechSupport': String,
                'StreamingTV': String,
                'StreamingMovies': String,
                'Contract': String,
                'PaperlessBilling': String,
                'PaymentMethod': String
            }
        }
    
    def get_feature_fields(self):
        """Generate Feast Fields from schema configuration"""
        fields = []
        for feature_type in self.schema_config.values():
            for name, dtype in feature_type.items():
                fields.append(Field(name=name, dtype=dtype))
        return fields

    def get_feature_names(self):
        """Get feature names by category"""
        return {
            'numerical': list(self.schema_config['numerical_features'].keys()),
            'categorical': list(self.schema_config['categorical_features'].keys())
        }
    
    def get_feature_refs(self, entity_id: str):
        """Generate feature references for feature store retrieval"""
        feature_refs = []
        for feature_type in self.schema_config.values():
            for feature_name in feature_type.keys():
                feature_refs.append(f"{entity_id}_features:{feature_name}")
        return feature_refs


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    feature_store_repo_path = "feature_repo"

class DataTransformation:
    def __init__(self):
        try:
            self.data_transformation_config = DataTransformationConfig()
            
            # Get absolute path and create directory structure
            repo_path = os.path.abspath(self.data_transformation_config.feature_store_repo_path)
            os.makedirs(os.path.join(repo_path, "data"), exist_ok=True)
            
            # Create feature store yaml configuration
            feature_store_yaml_path = os.path.join(repo_path, "feature_store.yaml")
            feature_store_yaml = """
project: behaviour_prediction
provider: local
registry: data/registry.db
online_store:
    type: sqlite
offline_store:
    type: file
entity_key_serialization_version: 2
"""
            # Write configuration file
            with open(feature_store_yaml_path, 'w') as f:
                f.write(feature_store_yaml)

            logging.info(f"Created feature store configuration at {feature_store_yaml_path}")

            # Verify the configuration file content
            with open(feature_store_yaml_path, 'r') as f:
                logging.info(f"Configuration file content:\n{f.read()}")
            
            # Initialize feature store
            self.feature_store = FeatureStore(repo_path=repo_path)
            logging.info("Feature store initialized successfully")
            
            self.schema = FeatureSchema()
            self.numerical_columns = self.schema.get_feature_names()['numerical']
            self.categorical_columns = self.schema.get_feature_names()['categorical']
            
            self.target_column = 'Churn'
            
        except Exception as e:
            raise CustomException(e, sys)
    def get_data_transformer_object(self):
        """
        Create preprocessing object for numerical and categorical columns
        """
        try:
            numeric_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot', OneHotEncoder(drop='first', sparse_output=False)),
                ]
            )
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numeric_pipeline, self.numerical_columns),
                    ('cat_pipeline', categorical_pipeline, self.categorical_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers_IQR(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Remove outliers using IQR method for specified columns
        """
        df_clean = df.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
            df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
        
        return df_clean
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")       

            # Remove customerID column
            train_df = train_df.drop('customerID', axis=1)
            test_df = test_df.drop('customerID', axis=1)
            
            # Handle missing values
            # train_df = train_df.dropna()
            # test_df = test_df.dropna()
            
            # Convert TotalCharges to numeric
            train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
            test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')
            
            # Handle outliers in numerical columns
            train_df = self.remove_outliers_IQR(train_df, self.numerical_columns)
            test_df = self.remove_outliers_IQR(test_df, self.numerical_columns)
                     
            # Encode target variable
            le = LabelEncoder()
            train_df[self.target_column] = le.fit_transform(train_df[self.target_column])
            test_df[self.target_column] = le.transform(test_df[self.target_column])
            
            # Split features and target
            input_feature_train_df = train_df.drop(columns=[self.target_column], axis=1)
            target_feature_train_df = train_df[self.target_column]
            
            input_feature_test_df = test_df.drop(columns=[self.target_column], axis=1)
            target_feature_test_df = test_df[self.target_column]
            
            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            # Transform the data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Starting feature store operations")

            # Push data to feature store
            self.push_features_to_store(train_df, "train")
            self.push_features_to_store(test_df, "test")

            logging.info("Pushed training & testing data to feature store")
            
            # Convert to numpy arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)

    def push_features_to_store(self, df: pd.DataFrame, entity_id: str):
        """
        Push features to the Feast feature store
        """
        try:
            # Add timestamp column if not present
            if 'event_timestamp' not in df.columns:
                df['event_timestamp'] = pd.Timestamp.now()
            
            # Add entity_id column if not present
            if 'entity_id' not in df.columns:
                df['entity_id'] = range(len(df))

            # Save data as parquet
            data_path = os.path.join(
                self.data_transformation_config.feature_store_repo_path,
                "data"
            )
            parquet_path = os.path.join(data_path, f"{entity_id}_features.parquet")
            
            # Ensure directory exists
            os.makedirs(data_path, exist_ok=True)
            
            # Save the parquet file
            df.to_parquet(parquet_path, index=False)

            # Define data source
            data_source = FileSource(
                path=f"data/{entity_id}_features.parquet",
                timestamp_field="event_timestamp"
            )

            # Define entity
            customer_entity = Entity(
                name="entity_id",
                value_type=ValueType.INT64,
                description="Customer ID"
            )

            # Define feature view with all features
            feature_view = FeatureView(
                name=f"{entity_id}_features",
                entities=[customer_entity],
                schema=self.schema.get_feature_fields(),
                source=data_source,
                online=True
            )

            # Apply to feature store
            self.feature_store.apply([customer_entity, feature_view])
            logging.info(f"Applied entity and feature view for {entity_id}")

            # Materialize features
            self.feature_store.materialize(
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now() + timedelta(days=1)
            )
            logging.info("Materialized features successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def retrieve_features_from_store(self, entity_id: str, df: pd.DataFrame):
        """
        Retrieve features from the feature store
        """
        try:
            feature_vector = self.feature_store.get_online_features(
                feature_refs=self.schema.get_feature_refs(entity_id),
                entity_rows=[{"entity_id": i} for i in range(len(df))]
            ).to_df()
            
            return feature_vector

        except Exception as e:
            raise CustomException(e, sys)

    