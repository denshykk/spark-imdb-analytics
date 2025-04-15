import os
import time

from src.analytics.denys_tykhonov import DenysTykhonovAnalytics
from src.analytics.ivan_dobrodieiev import IvanDobrodieievAnalytics
from src.analytics.ivan_pivtorak import IvanPivtorakAnalytics
from src.data.data_loader import ImdbDataLoader
from src.schemas.imdb_schemas import imdb_schemas
from src.utils.spark_utils import create_spark_session, print_schema_info, print_basic_stats


def test_spark_setup():
    """
    Tests if Spark is set up correctly by creating a simple DataFrame
    """
    spark = create_spark_session()

    test_data = [("Test", 1), ("Spark", 2), ("Setup", 3)]
    columns = ["word", "count"]

    print("Testing Spark setup with a simple DataFrame:")
    test_df = spark.createDataFrame(test_data, columns)
    test_df.show()

    return spark


def load_and_preview_datasets(spark):
    """
    Loads and previews all IMDB datasets

    Args:
        spark: SparkSession instance

    Returns:
        dict: Dictionary of loaded datasets
    """
    print("\n===== Loading IMDB Datasets =====")

    # Create data loader
    loader = ImdbDataLoader(spark)

    # Load all datasets with schemas
    datasets = loader.load_all_datasets(imdb_schemas)

    # Create temp views for SQL queries
    loader.create_temp_views()

    # Preview datasets
    for name, df in datasets.items():
        print_schema_info(df, name)

        # Calculate statistics for numeric columns (if any)
        numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ['int', 'double', 'float']]
        if numeric_cols:
            print_basic_stats(df, name, numeric_cols)

    return datasets


def run_all_analytics(spark, datasets):
    """
    Run all analytics implementations from team members

    Args:
        spark: SparkSession instance
        datasets: Dictionary of datasets
    """
    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # Create and run analytics for each team member
    print("\n===== Running All Team Members' Analytics =====")

    # Denys Tykhonov's analytics
    denys_analytics = DenysTykhonovAnalytics(spark, datasets)
    denys_analytics.run_all_analytics()

    # Ivan Dobrodieiev's analytics
    ivan_analytics = IvanDobrodieievAnalytics(spark, datasets)
    ivan_analytics.run_all_analytics()

    # Ivan Pivtorak's analytics
    ivan_pivtorak_analytics = IvanPivtorakAnalytics(spark, datasets)
    ivan_pivtorak_analytics.run_all_analytics()

    print("\n===== All Analytics Completed =====")


if __name__ == "__main__":
    start_time = time.time()

    print("Starting IMDB Data Analysis with Apache Spark")

    # Test Spark setup
    spark = test_spark_setup()

    # Load and preview datasets
    datasets = load_and_preview_datasets(spark)

    # Run all analytics
    run_all_analytics(spark, datasets)

    # Report execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")

    print("\nSpark IMDB Analytics completed successfully!")
