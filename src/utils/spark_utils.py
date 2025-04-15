import os

from pyspark.sql import SparkSession


def create_spark_session(app_name="IMDB Data Analysis", config=None):
    """
    Create and configure a Spark session

    Args:
        app_name: Name of the Spark application
        config: Additional configuration parameters

    Returns:
        SparkSession: Configured Spark session
    """
    builder = SparkSession.builder.appName(app_name)

    # Set default configuration
    builder = builder.config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse"))

    # Configure memory settings to avoid OOM errors
    builder = builder.config("spark.driver.memory", "4g")
    builder = builder.config("spark.executor.memory", "4g")
    builder = builder.config("spark.memory.offHeap.enabled", "true")
    builder = builder.config("spark.memory.offHeap.size", "2g")
    builder = builder.config("spark.sql.shuffle.partitions", "20")
    builder = builder.config("spark.default.parallelism", "20")

    # Add additional configuration if provided
    if config:
        for key, value in config.items():
            builder = builder.config(key, value)

    return builder.getOrCreate()


def write_to_csv(df, output_path, output_name, partition_by=None):
    """
    Write a DataFrame to CSV with memory optimizations

    Args:
        df: DataFrame to write
        output_path: Directory to write to
        output_name: Name of the output (without extension)
        partition_by: Column to partition by (optional)
    """
    path = os.path.join(output_path, output_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Write to CSV
    write_options = {
        "header": "true",
        "sep": ",",
        "encoding": "UTF-8"
    }

    # For large DataFrames, repartition to a smaller number to avoid OOM
    df_count = df.count()
    if df_count > 1000:
        num_partitions = min(20, df_count // 1000 + 1)
        df = df.repartition(num_partitions)

    try:
        if partition_by:
            df.write.csv(path, mode="overwrite", partitionBy=partition_by, **write_options)
        else:
            df.write.csv(path, mode="overwrite", **write_options)

        print(f"Results written to {path}")
    except Exception as e:
        print(f"Warning: Failed to write to CSV: {str(e)}")


def print_schema_info(df, dataset_name):
    """
    Print schema information for a DataFrame

    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset
    """
    print(f"\n===== Schema Information for {dataset_name} =====")
    df.printSchema()

    # Print number of rows and columns
    num_rows = df.count()
    num_cols = len(df.columns)
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")

    # Print sample data
    print("\nSample data:")
    df.show(5, truncate=False)


def print_basic_stats(df, dataset_name, numeric_cols=None):
    """
    Print basic statistics for a DataFrame

    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset
        numeric_cols: List of numeric columns to analyze (optional)
    """
    print(f"\n===== Basic Statistics for {dataset_name} =====")

    if numeric_cols:
        # Calculate statistics for specific numeric columns
        df.select(numeric_cols).describe().show()
    else:
        # Calculate statistics for all columns
        df.describe().show()
