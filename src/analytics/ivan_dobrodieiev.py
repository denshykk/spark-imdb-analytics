from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, avg, split, explode, rank, desc, dense_rank, count, lit, \
    approx_count_distinct
from pyspark.sql.window import Window

from src.utils.spark_utils import write_to_csv


class IvanDobrodieievAnalytics:
    """
    Contains 6 business questions and their implementation:
    1. Which directors have directed the most movies in a specific genre? (Filter, Join, Group By)
    2. Which directors and writers collaborate most frequently? (Join, Group By)
    3. What are the top-ranked movies within each genre? (Filter, Join, Window function)
    4. Which genres have the highest average runtime? (Filter, Window function)
    5. What is the distribution of adult vs non-adult movie titles by year? (Filter, Group By)
    6. Which job categories are most common across all movies? (Group By)
    """

    def __init__(self, spark: SparkSession, datasets: dict, output_path: str = "results/ivan_dobrodieiev"):
        self.spark = spark
        self.datasets = datasets
        self.output_path = output_path

    def directors_most_movies_in_genre(self):
        """
        Business Question 1: Which directors have directed the most movies in a specific genre?
        Returns:
            DataFrame: Directors with the most movies in a specific genre
        """
        # Get the basics dataset containing genres and movie titles
        basics_df = self.datasets["title.basics"]
        crew_df = self.datasets["title.crew"]
        names_df = self.datasets["name.basics"]

        # Filter for movies with non-null genres
        movie_df = basics_df.filter((col("titleType") == "movie") & (col("genres").isNotNull())).withColumn(
            "genres", split(col("genres"), ","))

        # Filter directors that are not null
        crew_df = crew_df.filter(col("directors").isNotNull())
        crew_df = crew_df.withColumnRenamed("tconst", "crew_tconst")

        # Join movies with crew info
        joined_df = movie_df.join(crew_df, movie_df["tconst"] == crew_df["crew_tconst"]).select(
            "tconst", "genres", "directors").withColumn("directors", split(col("directors"), ",")).withColumn(
            "director", col("directors")[0])

        # Group by full genre list and director
        grouped_df = joined_df.groupBy("genres", "director").count().withColumnRenamed("count", "movie_count")

        # Join with names to get director names
        result_df = grouped_df.join(names_df, grouped_df["director"] == names_df["nconst"], "left").select(
            col("genres")[0], col("primaryName").alias("director"), col("movie_count")).orderBy(
            col("movie_count").desc()).limit(20)

        write_to_csv(result_df, self.output_path, "directors_most_movies_in_genre")

        return result_df

    def directors_and_writers_collaboration(self):
        """
        Business Question: Which directors and writers collaborate most frequently?
        Returns:
            DataFrame: Directors and writers who collaborate most often
        """
        basics_df = self.datasets["title.basics"]
        crew_df = self.datasets["title.crew"]
        names_df = self.datasets["name.basics"]

        # Filter out rows with null values in directors or writers
        crew_df = crew_df.filter(col("directors").isNotNull() & col("writers").isNotNull())
        basics_df = basics_df.filter(col("startYear").isNotNull())

        # Filter for movies and join with crew
        movie_data = basics_df.filter(col("titleType") == "movie")
        movie_crew = movie_data.join(crew_df, movie_data["tconst"] == crew_df["tconst"])

        # Join the crew data with names_df to get the director and writer names
        directors_with_names = movie_crew \
            .join(names_df.alias("directors_df"), movie_crew["directors"] == col("directors_df.nconst"), "left") \
            .join(names_df.alias("writers_df"), movie_crew["writers"] == col("writers_df.nconst"), "left")

        directors_with_names = directors_with_names.filter(
            (col("directors_df.primaryName").isNotNull()) & (col("writers_df.primaryName").isNotNull()) &
            (col("directors_df.primaryName") != col("writers_df.primaryName"))
        )

        # Group by director and writer name, and count the number of movies they worked together on
        director_writer_collaboration = directors_with_names.groupBy("directors_df.primaryName",
                                                                     "writers_df.primaryName") \
            .count()

        # Rename columns for better readability
        result_df = director_writer_collaboration.select(
            col("directors_df.primaryName").alias("director_name"),
            col("writers_df.primaryName").alias("writer_name"),
            col("count").alias("collaboration_count")
        )

        # Order by the number of collaborations in descending order
        result_df = result_df.orderBy(col("collaboration_count").desc()).limit(20)

        write_to_csv(result_df, self.output_path, "directors_and_writers_collaboration")

        return result_df

    def top_movies_by_genre(self):
        """
        Business Question: What are the top-ranked movies within each genre?
        Returns:
            DataFrame: Top 5 movies in each genre by vote count
        """

        basics_df = self.datasets["title.basics"]
        ratings_df = self.datasets["title.ratings"]

        # Filter for movies with valid genres and join with ratings
        movies_with_ratings = basics_df.filter((col("titleType") == "movie") &
                                               (col("genres").isNotNull())).join(
            ratings_df, basics_df["tconst"] == ratings_df["tconst"]).filter(col("numVotes") >= 1000)

        # Explode the genres to create a row for each genre
        movies_by_genre = movies_with_ratings.withColumn("genre", explode(split(col("genres"), ","))).select(
            "primaryTitle", "startYear", "genre", "averageRating", "numVotes")

        # Create a window function to rank movies within each genre by vote count
        genre_window = Window.partitionBy("genre").orderBy(col("numVotes").desc())

        # Apply ranking
        ranked_movies = movies_by_genre.withColumn("genre_rank", rank().over(genre_window))

        # Filter for just the top 5 movies in each genre
        top_movies = ranked_movies.filter(col("genre_rank") <= 5)

        # Final result ordered by genre and rank
        result_df = top_movies.orderBy("genre", "genre_rank")

        # Write to CSV
        write_to_csv(result_df, self.output_path, "top_movies_by_genre")

        return result_df

    def genres_runtime_analysis(self):
        """
        Business Question: Which genres have the highest average runtime?
        Returns:
            DataFrame: Genres with runtime metrics including ranks and comparisons
        """
        basics_df = self.datasets["title.basics"]

        # Filter for movies with valid runtime and genres
        movie_data = basics_df.filter((col("titleType") == "movie") & (col("runtimeMinutes").isNotNull()) &
                                      (col("genres").isNotNull()) & (col("runtimeMinutes") > 0) & (
                                              col("runtimeMinutes") < 1000))

        # Explode genres to get one row per genre
        exploded_genres = movie_data.withColumn("genre", explode(split(col("genres"), ",")))

        # Calculate basic stats per genre
        genre_stats = exploded_genres.groupBy("genre").agg(
            avg("runtimeMinutes").cast("int").alias("avg_runtime"),
            count("*").alias("movie_count"))

        # Calculate global runtime average
        global_avg_runtime_val = genre_stats.select(avg("avg_runtime")).first()[0]

        # Add to genre_stats
        genre_stats = genre_stats.withColumn(
            "global_avg_runtime", lit(global_avg_runtime_val).cast("int")
        )

        # Window specs to calculate global metrics
        runtime_window = Window.orderBy(desc("avg_runtime"))
        count_window = Window.orderBy(desc("movie_count"))

        # Add window function columns
        result_df = genre_stats.withColumn(
            "runtime_rank", dense_rank().over(runtime_window)).withColumn(
            "popularity_rank", dense_rank().over(count_window)).withColumn(
            "diff_from_global_avg", col("avg_runtime") - col("global_avg_runtime")).withColumn(
            "percentage_diff",
            ((col("avg_runtime") - col("global_avg_runtime")) * 100 / col("global_avg_runtime")).cast("int"))

        # Order by average runtime descending
        result_df = result_df.orderBy(desc("avg_runtime"))

        # Write results to CSV
        write_to_csv(result_df, self.output_path, "genres_runtime_analysis")

        return result_df

    def adult_vs_non_adult_distribution(self):
        """
        Business Question 5: What is the distribution of adult vs non-adult movies titles by year?
        Returns:
            DataFrame: Distribution of adult vs non-adult movie titles by year
        """
        basics_df = self.datasets["title.basics"]

        # Filter movies by adult type
        adult_movies = basics_df.filter(col("isAdult") == 1)
        non_adult_movies = basics_df.filter(col("isAdult") == 0)

        # Group by year and count the adult and non-adult movies
        adult_count = adult_movies.groupBy("startYear").count().withColumnRenamed("count", "adult_count")
        non_adult_count = non_adult_movies.groupBy("startYear").count().withColumnRenamed("count", "non_adult_count")

        # Join the results to get a combined distribution
        result_df = adult_count.join(
            non_adult_count, "startYear", "outer"
        ).fillna(0).filter(col("startYear") <= 2025).orderBy(col("startYear").desc())

        write_to_csv(result_df, self.output_path, "adult_vs_non_adult_distribution")

        return result_df

    def job_categories_count(self):
        """
        Business Question: Which job categories are most common across all movies?
        Returns:
            Count of job categories by number of movies
        """
        principals_df = self.datasets["title.principals"]

        # Filter rows where the 'category' is not null
        job_categories_df = principals_df.filter(principals_df["category"].isNotNull())

        # Use approx_count_distinct to speed up counting
        result_df = job_categories_df.groupBy("category").agg(
            approx_count_distinct("tconst").alias("distinct_movies")
        ).orderBy(col("distinct_movies").desc())

        write_to_csv(result_df, self.output_path, "job_categories_count")

        return result_df

    def run_all_analytics(self):
        self.directors_most_movies_in_genre().show(20, truncate=False)
        self.directors_and_writers_collaboration().show(20, truncate=False)
        self.top_movies_by_genre().show(20, truncate=False)
        self.genres_runtime_analysis().show(20, truncate=False)
        self.adult_vs_non_adult_distribution().show(20, truncate=False)
        self.job_categories_count().show(20, truncate=False)
