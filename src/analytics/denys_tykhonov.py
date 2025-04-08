import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, avg
from pyspark.sql.functions import when
from pyspark.sql.window import Window

from src.utils.spark_utils import write_to_csv


class DenysTykhonovAnalytics:
    """
    Analytics implementation by Denys Tykhonov
    Contains 6 business questions and their implementation:
    1. What are the top 10 highest-rated movies with at least 10,000 votes? (Filter, Join)
    2. Which actors have appeared in the most TV series episodes? (Filter, Join)
    3. What is the average rating by genre for movies released after 2010? (Filter, Join, Group By)
    4. Which directors have improved their average movie ratings over time? (Filter, Join, Group By, Window Function)
    5. What are the most common languages for non-English movies? (Filter, Join, Group By)
    6. How has the average runtime of movies changed over decades? (Group By, Window)
    """

    def __init__(self, spark: SparkSession, datasets: dict, output_path: str = "results/denys"):
        """
        Initialize with a Spark session and datasets

        Args:
            spark: SparkSession instance
            datasets: Dictionary of datasets
            output_path: Path to write results
        """
        self.spark = spark
        self.datasets = datasets
        self.output_path = output_path

    def top_rated_movies(self):
        """
        Business Question 1: What are the top 10 highest-rated movies with at least 10,000 votes?

        Uses: JOIN & FILTER operation

        Returns:
            DataFrame: Top 10 highest-rated movies
        """
        # Get necessary datasets
        basics_df = self.datasets["title.basics"]
        ratings_df = self.datasets["title.ratings"]

        # Join and filter for movies with high vote counts
        result_df = basics_df.join(
            ratings_df,
            basics_df["tconst"] == ratings_df["tconst"]
        ).filter(
            (col("titleType") == "movie") &
            (col("numVotes") >= 10000)
        ).select(
            basics_df["tconst"],
            col("primaryTitle"),
            col("originalTitle"),
            col("startYear"),
            col("genres"),
            col("averageRating"),
            col("numVotes")
        ).orderBy(
            col("averageRating").desc()
        ).limit(10)

        write_to_csv(result_df, self.output_path, "top_rated_movies")

        return result_df

    def actors_in_most_tv_episodes(self):
        """
        Business Question 2: Which actors have appeared in the most TV series episodes?

        Uses: FILTER & JOIN operation

        Returns:
            DataFrame: Top 20 actors by TV episode count
        """
        name_df = self.datasets["name.basics"]
        basics_df = self.datasets["title.basics"]
        principals_df = self.datasets["title.principals"]

        # Filter by title type equals tv episode
        tv_episodes = basics_df.filter(col("titleType") == "tvEpisode").select("tconst")

        # Filter principals for actors/actresses only
        actors_principals = principals_df.filter(
            col("category").isin("actor", "actress")
        ).select("tconst", "nconst")

        # Filter names for just the columns we need
        name_basics = name_df.select("nconst", "primaryName")

        # First find actors in TV episodes
        actors_in_episodes = actors_principals.join(
            tv_episodes,
            actors_principals["tconst"] == tv_episodes["tconst"],
            "inner"
        ).select(actors_principals["nconst"])

        # Count episodes per actor
        episode_counts = actors_in_episodes.groupBy("nconst").count().withColumnRenamed("count", "episode_count")

        # Join with name info only for the top actors
        top_actors = episode_counts.orderBy(col("episode_count").desc()).limit(50)

        # Final join to get names
        result_df = top_actors.join(
            name_basics,
            top_actors["nconst"] == name_basics["nconst"],
            "inner"
        ).select(
            name_basics["nconst"],
            col("primaryName"),
            col("episode_count")
        ).orderBy(
            col("episode_count").desc()
        ).limit(20)

        result_df.show(truncate=False)

        try:
            write_to_csv(result_df, self.output_path, "actors_in_most_tv_episodes")
        except Exception as e:
            print(f"Warning: Could not write to CSV due to: {str(e)}")
            print("Continuing with analysis...")

        return result_df

    def avg_rating_by_genre(self):
        """
        Business Question 3: What is the average rating by genre for movies released after 2010?

        Uses: FILTER, JOIN & GROUP BY operation

        Returns:
            DataFrame: Average ratings by genre
        """
        basics_df = self.datasets["title.basics"]
        ratings_df = self.datasets["title.ratings"]

        # Process the genres column: convert string to array first, then explode
        from pyspark.sql.functions import split, explode

        # First convert the string genres to arrays
        movies_with_genres = basics_df.filter(
            (col("titleType") == "movie") &
            (col("startYear") > 2010) &
            (col("genres").isNotNull()) &
            (col("genres") != "\\N")
        ).withColumn(
            "genres_array",
            split(col("genres"), ",")
        )

        # Then explode the array
        exploded_genres = movies_with_genres.select(
            col("tconst"),
            col("primaryTitle"),
            col("startYear"),
            explode(col("genres_array")).alias("genre")
        )

        # Join with ratings and calculate average by genre
        result_df = exploded_genres.join(
            ratings_df,
            exploded_genres["tconst"] == ratings_df["tconst"]
        ).groupBy(
            col("genre")
        ).agg(
            avg("averageRating").alias("avg_rating"),
            count("*").alias("movie_count")
        ).filter(
            col("movie_count") >= 10  # Ensure we have enough movies for meaningful averages
        ).orderBy(
            col("avg_rating").desc()
        )

        try:
            write_to_csv(result_df, self.output_path, "avg_rating_by_genre")
        except Exception as e:
            print(f"Warning: Could not write to CSV due to: {str(e)}")
            print("Continuing with analysis...")

        return result_df

    def directors_with_improving_ratings(self):
        """
        Business Question 4: Which directors have improved their average movie ratings over time?

        Uses: FILTER, JOIN, GROUP BY & WINDOW function

        Returns:
            DataFrame: Directors with improving ratings
        """
        name_df = self.datasets["name.basics"]
        basics_df = self.datasets["title.basics"]
        ratings_df = self.datasets["title.ratings"]
        crew_df = self.datasets["title.crew"]

        # Process directors string to array first
        from pyspark.sql.functions import split, explode, lag

        # Find movies with directors
        directors_df = crew_df.filter(
            (col("directors").isNotNull()) &
            (col("directors") != "\\N")
        ).withColumn(
            "directors_array",
            split(col("directors"), ",")
        ).select(
            col("tconst"),
            explode(col("directors_array")).alias("director_id")
        )

        # Join with titles, ratings, and names
        joined_df = directors_df.join(
            basics_df,
            directors_df["tconst"] == basics_df["tconst"]
        ).join(
            ratings_df,
            basics_df["tconst"] == ratings_df["tconst"]
        ).join(
            name_df,
            directors_df["director_id"] == name_df["nconst"]
        ).filter(
            (col("titleType") == "movie") &
            (col("numVotes") >= 1000)  # Ensure enough votes for meaningful ratings
        ).select(
            col("director_id"),
            col("primaryName").alias("director_name"),
            basics_df["tconst"],
            col("primaryTitle"),
            col("startYear"),
            col("averageRating")
        )

        # Define window specifications
        window_spec = Window.partitionBy("director_id").orderBy("startYear")

        # Calculate average rating by year and the trend (difference from previous year)
        director_trends = joined_df.groupBy(
            col("director_id"),
            col("director_name"),
            col("startYear")
        ).agg(
            avg("averageRating").alias("avg_rating"),
            count("*").alias("movie_count")
        ).withColumn(
            "prev_year_avg",
            lag("avg_rating", 1).over(window_spec)
        ).withColumn(
            "rating_trend",
            col("avg_rating") - col("prev_year_avg")
        ).filter(
            col("prev_year_avg").isNotNull()
        )

        # Find directors with consistently improving ratings
        result_df = director_trends.groupBy(
            col("director_id"),
            col("director_name")
        ).agg(
            avg("rating_trend").alias("avg_rating_trend"),
            count("*").alias("years_with_movies")
        ).filter(
            (col("avg_rating_trend") > 0) &  # Positive trend
            (col("years_with_movies") >= 3)  # At least 3 years of data
        ).orderBy(
            col("avg_rating_trend").desc()
        ).limit(20)

        try:
            write_to_csv(result_df, self.output_path, "directors_with_improving_ratings")
        except Exception as e:
            print(f"Warning: Could not write to CSV due to: {str(e)}")
            print("Continuing with analysis...")

        return result_df

    def most_common_non_english_languages(self):
        """
        Business Question 5: What are the most common languages for non-English movies?

        Uses: FILTER, JOIN & GROUP BY operation

        Returns:
            DataFrame: Top languages by movie count
        """
        basics_df = self.datasets["title.basics"]
        akas_df = self.datasets["title.akas"]

        # Filter for movies and join with language information
        result_df = basics_df.filter(
            col("titleType") == "movie"
        ).join(
            akas_df,
            basics_df["tconst"] == akas_df["titleId"]
        ).filter(
            (col("language").isNotNull()) &
            (col("language") != "\\N") &
            (col("language") != "en")
        ).groupBy(
            col("language")
        ).agg(
            count("*").alias("movie_count")
        ).orderBy(
            col("movie_count").desc()
        ).limit(20)

        write_to_csv(result_df, self.output_path, "most_common_non_english_languages")

        return result_df

    def movie_runtime_by_decade(self):
        """
        Business Question 6: How has the average runtime of movies changed over decades?

        Uses: GROUP BY operation with window function

        Returns:
            DataFrame: Average runtime by decade
        """
        basics_df = self.datasets["title.basics"]

        movies_with_runtime = basics_df.filter(
            (col("titleType") == "movie")
        ).withColumn(
            "runtimeMinutes", col("runtimeMinutes").cast("int")
        ).withColumn(
            "startYear", col("startYear").cast("int")
        ).filter(
            (col("runtimeMinutes").isNotNull()) &
            (col("runtimeMinutes") > 0) &
            (col("startYear").isNotNull()) &
            (col("startYear") > 1900)
        )

        # Calculate decade from startYear
        movies_with_decade = movies_with_runtime.withColumn(
            "decade",
            (col("startYear") / 10).cast("int") * 10
        )

        # Group by decade and calculate average runtime
        decade_stats = movies_with_decade.groupBy(
            col("decade")
        ).agg(
            avg("runtimeMinutes").alias("avg_runtime"),
            count("*").alias("movie_count")
        ).filter(
            col("decade").isNotNull()
        ).orderBy(
            col("decade")
        )

        # Calculate runtime trend using window function
        from pyspark.sql.functions import lag
        window_spec = Window.orderBy("decade")

        result_df = decade_stats.withColumn(
            "prev_decade_runtime",
            lag("avg_runtime", 1).over(window_spec)
        ).withColumn(
            "runtime_change",
            col("avg_runtime") - col("prev_decade_runtime")
        ).withColumn(
            "runtime_change_percent",
            when(col("prev_decade_runtime").isNotNull(),
                 (col("runtime_change") / col("prev_decade_runtime") * 100)
                 ).otherwise(None)
        )

        try:
            write_to_csv(result_df, self.output_path, "movie_runtime_by_decade")
        except Exception as e:
            print(f"Warning: Could not write to CSV due to: {str(e)}")
            print("Continuing with analysis...")

        return result_df

    def run_all_analytics(self):
        """
        Run all business questions analytics
        """
        print("\n===== Running Denys Tykhonov's Analytics =====")

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        try:
            print("\n1. Finding top rated movies...")
            self.top_rated_movies().show(10, truncate=False)
        except Exception as e:
            print(f"Error in top_rated_movies: {str(e)}")

        try:
            print("\n2. Finding actors who appeared in most TV episodes...")
            self.actors_in_most_tv_episodes().show(10, truncate=False)
        except Exception as e:
            print(f"Error in actors_in_most_tv_episodes: {str(e)}")

        try:
            print("\n3. Calculating average rating by genre for recent movies...")
            self.avg_rating_by_genre().show(10, truncate=False)
        except Exception as e:
            print(f"Error in avg_rating_by_genre: {str(e)}")

        try:
            print("\n4. Finding directors with improving ratings over time...")
            self.directors_with_improving_ratings().show(10, truncate=False)
        except Exception as e:
            print(f"Error in directors_with_improving_ratings: {str(e)}")

        try:
            print("\n5. Finding most common non-English languages in movies...")
            self.most_common_non_english_languages().show(10, truncate=False)
        except Exception as e:
            print(f"Error in most_common_non_english_languages: {str(e)}")

        try:
            print("\n6. Analyzing how movie runtime has changed by decade...")
            self.movie_runtime_by_decade().show(truncate=False)
        except Exception as e:
            print(f"Error in movie_runtime_by_decade: {str(e)}")

        print(f"\nFinished Denys Tykhonov's analytics")
