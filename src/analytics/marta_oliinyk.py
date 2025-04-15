import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, avg, count, dense_rank, desc, size, collect_set, floor, lag, \
    row_number
from pyspark.sql.window import Window
from src.utils.spark_utils import write_to_csv


class MartaOliinykAnalytics:
    """
    Contains 6 new business questions and their implementation:
    1. Who are the most frequent actors in a specific genre? (Filter, Join, Explode, Group By)
    2. What are the trends in average movie ratings by decade? (Filter, Group By)
    3. What are the top-rated movies by country of origin? (Join, Window Function)
    4. Which actors have acted in the most diverse set of genres? (Join, Explode, Aggregation)
    5. Which writers have written the most top-rated movies (rating > 8.5)? (Filter, Join, Group By)
    6. Which genres have seen the biggest increase in production volume over the last 3 decades?
    """

    def __init__(self, spark: SparkSession, datasets: dict, output_path: str = "results/marta"):
        self.spark = spark
        self.datasets = datasets
        self.output_path = output_path

    def most_frequent_actors_by_genre(self):
        """
        Business Question 1: Who are the most frequent actors in a specific genre?
        """
        basics_df = self.datasets["title.basics"]
        principals_df = self.datasets["title.principals"]
        names_df = self.datasets["name.basics"]

        movies_df = basics_df.filter((col("titleType") == "movie") & col("genres").isNotNull())
        actors_df = principals_df.filter(col("category").isin("actor", "actress"))

        genre_exploded = movies_df.withColumn("genre", explode(split(col("genres"), ",")))

        joined_df = genre_exploded.join(actors_df, "tconst") \
            .join(names_df, "nconst") \
            .groupBy("genre", "primaryName") \
            .count().withColumnRenamed("count", "appearance_count") \
            .orderBy(desc("appearance_count"))

        result_df = joined_df.withColumn("rank", dense_rank().over(
            Window.partitionBy("genre").orderBy(desc("genre"), desc("appearance_count")))).filter(col("rank") <= 1)

        write_to_csv(result_df, self.output_path, "most_frequent_actors_by_genre")

        return result_df

    def average_rating_by_decade(self):
        """
        Business Question 2: What are the average movie ratings by decade?
        """
        basics_df = self.datasets["title.basics"]
        ratings_df = self.datasets["title.ratings"]

        joined_df = basics_df.filter(col("startYear").isNotNull() & col("startYear").rlike("^[0-9]{4}$")) \
            .join(ratings_df, "tconst")

        decade_df = joined_df.withColumn("decade", (col("startYear").cast("int") / 10).cast("int") * 10)

        result_df = decade_df.groupBy("decade").agg(
            avg("averageRating").alias("avg_rating"),
            count("*").alias("movie_count")
        ).orderBy("decade")

        write_to_csv(result_df, self.output_path, "average_rating_by_decade")

        return result_df

    def top_rated_by_region(self):
        """
        Business Question 3: What are the top-rated movies by region of origin?
        """
        basics_df = self.datasets["title.basics"]
        ratings_df = self.datasets["title.ratings"]
        akas_df = self.datasets["title.akas"]

        movies_df = basics_df.filter(
            (col("titleType") == "movie")
        )

        rated_movies_df = movies_df.join(ratings_df, "tconst") \
            .filter(col("numVotes") >= 1000)

        akas_filtered = akas_df.filter(
            col("region").isNotNull()
        ).select("titleId", "region").dropDuplicates(["titleId", "region"])

        joined_df = rated_movies_df.join(
            akas_filtered, rated_movies_df["tconst"] == akas_filtered["titleId"]
        ).select(
            "primaryTitle", "startYear", "region", "averageRating", "numVotes"
        )

        region_window = Window.partitionBy("region").orderBy(col("averageRating").desc(), col("numVotes").desc())

        ranked_df = joined_df.withColumn("rank", row_number().over(region_window)) \
            .filter(col("rank") <= 3) \
            .orderBy("region", "rank")

        write_to_csv(ranked_df, self.output_path, "top_rated_by_country")

        return ranked_df

    def most_genre_diverse_actors(self):
        """
        Business Question 4: Which actors have acted in the most diverse set of genres?
        """
        basics_df = self.datasets["title.basics"]
        principals_df = self.datasets["title.principals"]
        names_df = self.datasets["name.basics"]

        movie_df = basics_df.filter((col("titleType") == "movie") & col("genres").isNotNull())
        actors_df = principals_df.filter(col("category").isin("actor", "actress"))

        genre_exploded = movie_df.withColumn("genre", explode(split(col("genres"), ",")))

        joined_df = genre_exploded.join(actors_df, "tconst") \
            .groupBy("nconst") \
            .agg(collect_set("genre").alias("genres")) \
            .withColumn("genre_count", size("genres"))

        result_df = joined_df.join(names_df, "nconst") \
            .select("primaryName", "genre_count") \
            .orderBy(desc("genre_count")) \
            .limit(10)

        write_to_csv(result_df, self.output_path, "most_genre_diverse_actors")

        return result_df

    def top_directors_by_high_rated_movies(self):
        """
        Business Question 5: Which directors have written the most top-rated movies (rating > 8.5)?
        """
        basics_df = self.datasets["title.basics"]
        ratings_df = self.datasets["title.ratings"]
        crew_df = self.datasets["title.crew"]
        names_df = self.datasets["name.basics"]

        high_rated_movies = ratings_df.filter(col("averageRating") >= 8.5)
        movie_info = basics_df.filter(col("titleType") == "movie") \
            .join(high_rated_movies, "tconst") \
            .join(crew_df, "tconst") \
            .filter(col("directors").isNotNull()) \
            .withColumn("director", explode(split(col("directors"), ",")))

        director_counts = movie_info.groupBy("director").count().withColumnRenamed("count", "top_movie_count")
        result_df = director_counts.join(names_df, director_counts["director"] == names_df["nconst"], "left") \
            .select("primaryName", "top_movie_count") \
            .orderBy(desc("top_movie_count")) \
            .limit(10)

        write_to_csv(result_df, self.output_path, "top_directors_by_high_rated_movies")

        return result_df

    def genre_growth_trend(self):
        """
        Business Question: Which genres have seen the biggest increase in production volume over the last 3 decades?
        """
        basics_df = self.datasets["title.basics"]

        filtered = basics_df.filter(
            (col("titleType") == "movie") &
            (col("genres").isNotNull()) &
            (col("startYear").isNotNull()) &
            (col("startYear") >= 1990)
        ).withColumn(
            "decade", (floor(col("startYear").cast("int") / 10) * 10).cast("int")
        ).withColumn(
            "genre", explode(split(col("genres"), ","))
        )

        genre_counts = filtered.groupBy("decade", "genre").agg(
            count("*").alias("movie_count")
        )

        window_spec = Window.partitionBy("genre").orderBy("decade")
        genre_counts = genre_counts.withColumn(
            "prev_decade_count", lag("movie_count").over(window_spec)
        ).withColumn(
            "growth", col("movie_count") - col("prev_decade_count")
        )

        result_df = genre_counts.orderBy(col("growth").desc()).limit(20)

        write_to_csv(result_df, self.output_path, "genre_production_growth")
        return result_df

    def run_all_analytics(self):
        """
        Run all business questions analytics
        """
        print("\n===== Running Marta Oliinyk's Analytics =====")

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        try:
            print("\n1. Finding most frequent actors in a specific genre...")
            self.most_frequent_actors_by_genre().show(10, truncate=False)
        except Exception as e:
            print(f"Error in most_frequent_actors_by_genre: {str(e)}")

        try:
            print("\n2. Calculating average movie ratings by decade...")
            self.average_rating_by_decade().show(10, truncate=False)
        except Exception as e:
            print(f"Error in average_rating_by_decade: {str(e)}")

        try:
            print("\n3. Finding the top-rated movies by region of origin...")
            self.top_rated_by_region().show(10, truncate=False)
        except Exception as e:
            print(f"Error in top_rated_by_country: {str(e)}")

        try:
            print("\n4. Finding actors that have acted in the most diverse set of genres...")
            self.most_genre_diverse_actors().show(10, truncate=False)
        except Exception as e:
            print(f"Error in most_genre_diverse_actors: {str(e)}")

        try:
            print("\n5. Finding directors with the most top-rated movies...")
            self.top_directors_by_high_rated_movies().show(10, truncate=False)
        except Exception as e:
            print(f"Error in top_directors_by_high_rated_movies: {str(e)}")

        try:
            print("\n6. Analyzing genres with biggest increase in production volume...")
            self.genre_growth_trend().show(10, truncate=False)
        except Exception as e:
            print(f"Error in genre_growth_trend: {str(e)}")

        print(f"\nFinished Marta Oliinyk's analytics")
