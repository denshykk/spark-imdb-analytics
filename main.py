import os

from pyspark.sql import SparkSession


def create_spark_session():
    """
    Creates and returns a Spark session
    """
    return SparkSession.builder \
        .appName("IMDB Data Analysis") \
        .config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse")) \
        .getOrCreate()


def test_spark_setup():
    """
    Tests if Spark is set up correctly by creating a simple DataFrame
    """
    spark = create_spark_session()

    test_data = [("Test", 1), ("Spark", 2), ("Setup", 3)]
    columns = ["word", "count"]
    test_df = spark.createDataFrame(test_data, columns)

    print("Testing Spark setup with a simple DataFrame:")
    test_df.show()

    return test_df


if __name__ == "__main__":
    print("Starting IMDB Data Analysis with Apache Spark")
    test_df = test_spark_setup()
    print("Spark setup successful!")
