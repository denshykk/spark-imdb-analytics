from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
import os


class ImdbDataLoader:
    """
    Class for loading and managing IMDB datasets
    """

    def __init__(self, spark: SparkSession, data_path: str = "imdb_data"):
        """
        Initialize the data loader with a Spark session and data path

        Args:
            spark: The SparkSession instance
            data_path: Path to the directory containing IMDB data files
        """
        self.spark = spark
        self.data_path = data_path
        self.datasets = {}

    def load_dataset(self, filename: str, schema: StructType = None):
        """
        Load a dataset from the data path

        Args:
            filename: Name of the file to load (without .tsv extension)
            schema: Optional schema for the dataset

        Returns:
            DataFrame: The loaded dataset
        """
        file_path = os.path.join(self.data_path, f"{filename}.tsv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        if schema:
            df = self.spark.read.csv(file_path, sep="\t", header=True, schema=schema, nullValue="\\N")
        else:
            df = self.spark.read.csv(file_path, sep="\t", header=True, inferSchema=True, nullValue="\\N")

        # Cache the dataset for reuse
        df.cache()

        # Store in the datasets dictionary
        self.datasets[filename] = df

        return df

    def load_all_datasets(self, schemas_dict=None):
        """
        Load all IMDB datasets

        Args:
            schemas_dict: Dictionary mapping dataset names to schemas

        Returns:
            dict: Dictionary of all loaded datasets
        """
        dataset_names = [
            "name.basics",
            "title.akas",
            "title.basics",
            "title.crew",
            "title.episode",
            "title.principals",
            "title.ratings"
        ]

        for name in dataset_names:
            schema = schemas_dict.get(name) if schemas_dict else None
            try:
                self.load_dataset(name, schema)
                print(f"Successfully loaded {name}")
            except Exception as e:
                print(f"Error loading {name}: {e}")

        return self.datasets

    def create_temp_views(self):
        """
        Create temporary views for all loaded datasets for SQL usage
        """
        for name, df in self.datasets.items():
            # Convert dots to underscores for valid table names
            view_name = name.replace(".", "_")
            df.createOrReplaceTempView(view_name)
            print(f"Created temp view: {view_name}")
