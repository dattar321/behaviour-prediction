# Data Ingestion Component

### Components
1. `DataIngestionConfig`
A configuration class managing file paths for:

- Raw data: `artifacts/data_ingestion/raw.csv`

- Training data: `artifacts/data_ingestion/train.csv`

- Testing data: `artifacts/data_ingestion/test.csv`


## Functionality
### Data Processing Steps

1. `Read Raw Data`:
- Reads behavior_cleandata.csv from data-source directory
- Uses Pandas for data loading


2.Data Splitting
- Splits data into:
   - Training set: 80%
   - Testing set: 20%

3. `Data Saving`:

- Creates necessary directories
- Saves raw, train, and test datasets as CSV files


## DVC After Data Ingestion 
Once the data ingestion process is complete, you can use DVC (Data Version Control) to version the data files just like you version code in Git. Here's how you can proceed:

1. Initialize DVC:

```bash
dvc init
git add .dvc
git commit -m "Initialize DVC"
```
2. Add Data Files to DVC:
Use DVC to track the ingested data files (train, test, and raw data):

```bash
dvc add artifacts/data_ingestion/raw.csv
dvc add artifacts/data_ingestion/train.csv
dvc add artifacts/data_ingestion/test.csv
```
3. Commit Changes to Git:
Add the DVC metadata and updated .gitignore file to Git:

```bash
git add artifacts/data_ingestion/*.dvc .gitignore
git commit -m "Track ingested data with DVC"
```
4. Push:
```
bash
git add .dvc/config
git commit -m "Configured remote storage for DVC"
```
---

## Data Transformation Component

#### 1. **Overview**
- A feature schema definition for numerical and categorical features.
- A `DataTransformation` class to preprocess data, remove outliers, and manage features using Feast.
- Methods for pushing features into the store, retrieving them for training or inference, and materializing them.

#### 2. **Classes and Methods**

##### **`FeatureSchema` Class**
Defines the schema for numerical and categorical features in the dataset.

**Attributes:**
- `schema_config`: A dictionary specifying:
  - `numerical_features`: Features treated as numerical with their data types.
  - `categorical_features`: Features treated as categorical with their data types.

**Methods:**
- `get_feature_fields()`: Generates a list of Feast `Field` objects from the schema for registration in the feature store.
- `get_feature_names()`: Returns feature names grouped by category (numerical and categorical).
- `get_feature_refs(entity_id: str)`: Generates feature references (`<entity_id>_features:<feature_name>`) for retrieving features from the store.

##### **`DataTransformationConfig` Class**
Stores configuration paths for data transformation.

**Attributes:**
- `preprocessor_obj_file_path`: Path to save the preprocessing object (`preprocessor.pkl`).
- `feature_store_repo_path`: Path to the Feast feature repository (`feature_repo`).

##### **`DataTransformation` Class**
Handles data transformation, outlier removal, preprocessing, and feature management with Feast.

**Initialization (`__init__`):**
- Creates directory structures for the feature store.
- Generates the `feature_store.yaml` configuration file.
- Initializes a local feature store with Feast.
- Loads feature schema and separates numerical and categorical feature names.
- Defines the target column.

**Key Methods:**
- `get_data_transformer_object()`: Creates a preprocessing pipeline:
  - **Numerical Pipeline**:
    - Imputes missing values with the median.
    - Scales values using `StandardScaler`.
  - **Categorical Pipeline**:
    - Imputes missing values with the most frequent value.
    - Applies one-hot encoding.
  - Returns a `ColumnTransformer` combining both pipelines.

- `remove_outliers_IQR(df: pd.DataFrame, columns: list)`: Removes outliers from specified columns using the IQR (Interquartile Range) method:
  - Values outside `[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]` are replaced with boundary values.

- `initiate_data_transformation(train_path, test_path)`: Main method to preprocess the data and prepare it for ML pipelines:
  - **Step 1**: Load train and test datasets from CSV files.
  - **Step 2**: Drop irrelevant columns (e.g., `customerID`).
  - **Step 3**: Handle missing values and convert `TotalCharges` to numeric.
  - **Step 4**: Remove outliers from numerical columns.
  - **Step 5**: Encode the target variable (`Churn`) using `LabelEncoder`.
  - **Step 6**: Apply the preprocessing pipeline on features.
  - **Step 7**: Push processed features to the Feast feature store.
  - **Step 8**: Save the preprocessing pipeline for reuse.
  - Returns processed train and test arrays, along with the preprocessing object path.

- `push_features_to_store(df: pd.DataFrame, entity_id: str)`: Pushes features to the Feast feature store:
  - **Step 1**: Adds a timestamp and `entity_id` column if missing.
  - **Step 2**: Saves the dataset as a Parquet file in `feature_repo/data`.
  - **Step 3**: Defines Feast objects:
    - `Entity`: Represents the `entity_id` column.
    - `FeatureView`: Groups the features in the dataset.
  - **Step 4**: Applies the entity and feature view to the feature store.
  - **Step 5**: Materializes features for retrieval in the online and offline stores.

- `retrieve_features_from_store(entity_id: str, df: pd.DataFrame)`: Retrieves features from the Feast online store:
  - Queries the feature store for a feature vector based on `entity_id`.

#### 3. **Directory Structure**
The code organizes files and configurations as follows:

```bash
project-root/
│
├── artifacts/
│   └── data_transformation/
│       └── preprocessor.pkl  
│
├── feature_repo/             
│   ├── feature_store.yaml    
│   ├── data/                 
│       ├── train_features.parquet
│       └── test_features.parquet
│
└── logs/                     
```

#### 4. **Feature Store Configuration (`feature_store.yaml`)**
Defines the setup for the Feast feature store:

```yaml
project: behaviour_prediction
provider: local
registry: data/registry.db
online_store:
    type: sqlite
offline_store:
    type: file
entity_key_serialization_version: 2
```

- **project**: Name of the Feast project.
- **provider**: Specifies the backend (local for file-based storage).
- **registry**: Stores metadata about entities and feature views.
- **online_store**: SQLite database for real-time feature retrieval.
- **offline_store**: File-based storage for batch feature retrieval.

#### 5. **Key Concepts**

- **Feature Schema**: Provides a structure for defining numerical and categorical features. Enables dynamic creation of feature views in Feast.
- **Feature Store**: Centralized storage for ML features. Separates feature management from the ML pipeline. Enables real-time and batch retrieval of features for inference.
#### 6. Feature Schema and Data Transformation with Feast
##### Directory Overview:

1. `artifacts/data_transformation/preprocessor.pkl`:
Stores the preprocessing object (ColumnTransformer) for transforming numerical and categorical features.

2. `feature_repo/data`:
Stores feature data as Parquet files (e.g., train_features.parquet, test_features.parquet).

3. `feature_repo/feature_store.yaml`:
The Feast configuration file, which defines the feature store's setup, such as provider (local), registry, and storage options.

---

## Model Trainer Component

### Key Components

#### **`ModelTrainingConfig`**

This configuration class holds settings for MLFlow and model paths.

##### **Attributes**:
- `train_model_file_path`: Path where the trained model will be saved.
- `mlflow_uri`: URI for the MLFlow tracking server.
- `experiment_name`: Name for the MLFlow experiment.

---

#### **`ModelTrainer`**

The `ModelTrainer` class encapsulates methods for training, evaluating, and logging machine learning models using MLFlow.

##### **Key Methods**:
- `__init__()`: Initializes MLFlow with the tracking URI and experiment name.
- `log_metrics()`: Logs evaluation metrics (accuracy, precision, recall, F1 score) to MLFlow.
- `train_model()`: Trains a model using Grid Search and logs metrics, best parameters, and feature importance.
- `initiate_model_trainer()`: Handles the complete training process for multiple models and logs the best model.
- `main()`: Orchestrates the entire pipeline, including data ingestion, transformation, and model training.

---

### Model Training Workflow

#### **Initialization**:
- MLFlow tracking URI and experiment name are set in the `ModelTrainingConfig`.
- A unique run name is generated for each training session.

#### **Model Training**:
- The pipeline trains multiple models (Logistic Regression, KNN, Random Forest, SVM, XGBoost, AdaBoost, and GradientBoosting) using a grid search for hyperparameter optimization.
- For each model, parameters are logged, the model is trained, and performance metrics (accuracy, precision, recall, F1 score) are calculated and logged to MLFlow.

#### **Best Model Selection**:
- After training, the best model is determined based on accuracy and saved locally in the specified `model.pkl` file.
- A model comparison plot is generated and logged.

#### **MLFlow Logging**:
- All metrics, parameters, and artifacts (model, plots) are logged to MLFlow for experiment tracking.
- Feature importance plots are also logged if available.

---

### Directory Structure

- `artifacts/`: Stores model and plot artifacts.
- `model_trainer/`: Contains model-related files (e.g., `model.pkl`).
- `logs/`: Contains logs for debugging and tracking.
- `mlruns/`: Directory where MLFlow stores experiment data by default.

---

## Training Pipeline 

### Training Workflow
Pipeline flow integrating data version control, feature engineering, and model tracking:

```mermaid
graph LR
    ingestion[Data Ingestion] --> transformation[Data Transformation]
    transformation --> model_trainer[Model Training]

```

## Prediction Pipeline 

- Create CustomData instance with customer features
- Convert to DataFrame using get_data_as_dataframe()
- Use PredictionPipeline().predict() to get model predictions
- The model trained earlier was saved as a pickle file and is loaded in the prediction pipeline to predict the results.