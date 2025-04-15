from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, avg, split, explode, rank, desc, dense_rank, count, lit, \
    approx_count_distinct, lag, floor
from pyspark.sql.window import Window

from src.utils.spark_utils import write_to_csv


class IvanPivtorakAnalytics:
    """
    Contains 6 business questions and their implementation:
    1. Which movies are most frequently mentioned in knownForTitles among actors? (Filter, Join, Group By)
    2. How has the average number of episodes per season in TV series changed from the 1990s to today? (Filter, Group By)
    3. What is the average gap between consecutive movie releases for the top 20 most prolific directors? (join, window function)
    4. What is the average number of votes per year for movies rated above 7.5? (filter, join, window function)
    5. Which movie titles have identical names and how many such versions exist? (group by, filter)
    6. What are the top 5 longest-running TV series by number of episodes? (filter, join, group by)
    """

    def __init__(self, spark: SparkSession, datasets: dict, output_path: str = "results/ivan_pivtorak"):
        self.spark = spark
        self.datasets = datasets
        self.output_path = output_path

    def most_mentioned_titles_by_actors(self):
        """
        Business Question: Which movies are most frequently mentioned in knownForTitles among actors?
        Returns:
            DataFrame: Top mentioned titles (movies) in knownForTitles field by actors
        """
        from pyspark.sql.functions import col, split, explode

        names_df = self.datasets["name.basics"]
        basics_df = self.datasets["title.basics"]

        actors_df = names_df.filter(
            (col("primaryProfession").contains("actor")) | (col("primaryProfession").contains("actress"))
        ).filter(col("knownForTitles").isNotNull())

        actors_df = actors_df.withColumn("knownForArray", split(col("knownForTitles"), ","))

        exploded_df = actors_df.select(explode(col("knownForArray")).alias("tconst"))

        counted_df = exploded_df.groupBy("tconst").count().withColumnRenamed("count", "mention_count")

        result_df = counted_df.join(
            basics_df.select("tconst", "primaryTitle", "startYear"),
            on="tconst",
            how="left"
        ).select("tconst", "primaryTitle", "startYear", "mention_count").orderBy(col("mention_count").desc()).limit(20)

        write_to_csv(result_df, self.output_path, "most_mentioned_titles_by_actors")

        return result_df

    def avg_episodes_per_season_over_time(self):
        """
        Business Question: How has the average number of episodes per season in TV series changed from the 1990s to today?
        Uses: Filter, Group By, Window Function
        Returns:
            DataFrame: Average number of episodes per season by year.
        """
        episodes_df = self.datasets["title.episode"]
        basics_df = self.datasets["title.basics"]

        joined_df = episodes_df.join(
            basics_df.select(col("tconst").alias("parentTconst"), "startYear"),
            on="parentTconst",
            how="left"
        )

        filtered_df = joined_df.filter(
            (col("startYear") >= 1990) &
            col("seasonNumber").isNotNull()
        )

        season_episodes = filtered_df.groupBy("startYear", "parentTconst", "seasonNumber") \
            .agg(count("tconst").alias("episode_count"))

        avg_per_year = season_episodes.groupBy("startYear") \
            .agg(avg("episode_count").alias("avg_episodes")) \
            .orderBy("startYear")

        write_to_csv(avg_per_year, self.output_path, "avg_episodes_per_season_over_time")

        return avg_per_year

    def average_gap_between_director_movies(self):
        """
        Business Question: What is the average gap between consecutive movie releases for the top 20 most prolific directors?
        Returns:
            DataFrame: Top 20 directors with the highest number of movies and their average release gap.
        """
        from pyspark.sql.functions import col, split, explode, lag, avg, count
        from pyspark.sql.window import Window

        basics_df = self.datasets["title.basics"]
        crew_df = self.datasets["title.crew"]
        names_df = self.datasets["name.basics"]

        movies_df = basics_df.filter(col("titleType") == "movie") \
            .filter(col("startYear").isNotNull())

        movie_directors_df = movies_df.join(
            crew_df.select("tconst", "directors"),
            on="tconst",
            how="inner"
        ).filter(col("directors").isNotNull())

        movie_directors_df = movie_directors_df.withColumn("director", explode(split(col("directors"), ",")))

        director_movies_df = movie_directors_df.select(
            "tconst", "startYear", "director"
        ).withColumn("startYear", col("startYear").cast("int"))

        window_spec = Window.partitionBy("director").orderBy("startYear")
        director_movies_df = director_movies_df.withColumn("prevYear", lag("startYear").over(window_spec))

        director_movies_df = director_movies_df.withColumn("yearGap", col("startYear") - col("prevYear"))

        director_stats_df = director_movies_df.groupBy("director") \
            .agg(
            avg("yearGap").alias("averageGap"),
            count("tconst").alias("movieCount")
        ).filter(col("averageGap").isNotNull())

        top_20_df = director_stats_df.orderBy(col("movieCount").desc()).limit(20)

        result_df = top_20_df.join(
            names_df.select("nconst", "primaryName"),
            top_20_df.director == names_df.nconst,
            how="left"
        ).select("director", "primaryName", "movieCount", "averageGap") \
            .orderBy(col("movieCount").desc(), col("averageGap"))

        write_to_csv(result_df, self.output_path, "top20_directors_average_gap")

        return result_df

    def avg_votes_per_year_high_rated(self):
        """
        Business Question: What is the average number of votes per year for movies rated above 7.5?
        Returns:
            DataFrame: Average number of votes for high-rated movies (rating > 7.5), grouped by release year,
            including cumulative average with Window function.
        """
        from pyspark.sql.functions import col, avg
        from pyspark.sql.window import Window

        basics_df = self.datasets["title.basics"]
        ratings_df = self.datasets["title.ratings"]

        joined_df = ratings_df.join(
            basics_df.select("tconst", "startYear"),
            on="tconst",
            how="inner"
        )

        filtered_df = joined_df.filter(
            (col("averageRating") > 7.5) & (col("startYear").isNotNull())
        )

        per_year_df = filtered_df.groupBy("startYear").agg(
            avg("numVotes").alias("avgVotes")
        )

        window_spec = Window.orderBy("startYear").rowsBetween(Window.unboundedPreceding, Window.currentRow)

        result_df = per_year_df.withColumn(
            "cumulativeAvgVotes", avg("avgVotes").over(window_spec)
        ).orderBy("startYear")

        write_to_csv(result_df, self.output_path, "avg_votes_per_year_high_rated")

        return result_df

    def count_identical_titles(self):
        """
        Business Question: Which movie titles have identical names (excluding episodes)
        and how many such versions exist?

        Returns:
            DataFrame: Movie titles with number of identical versions (excluding episodes)
        """
        from pyspark.sql.functions import count, col

        basics_df = self.datasets["title.basics"]
        episodes_df = self.datasets["title.episode"]

        non_episodes_df = basics_df.join(
            episodes_df.select("tconst"), on="tconst", how="left_anti"
        )

        result_df = non_episodes_df.groupBy("primaryTitle").agg(
            count("primaryTitle").alias("numVersions")
        ).filter(col("numVersions") > 1).orderBy(col("numVersions").desc())

        top_20_result_df = result_df.limit(20)

        write_to_csv(top_20_result_df, self.output_path, "count_identical_titles_no_episodes")

        return top_20_result_df

    def top_longest_tv_series_by_episodes(self):
        """
        Business Question: What are the top 5 longest-running TV series by number of episodes?

        Returns:
            DataFrame: Top 5 TV series with the most episodes
        """
        from pyspark.sql.functions import count, col

        episode_df = self.datasets["title.episode"]
        basics_df = self.datasets["title.basics"]

        tv_series_df = basics_df.filter(col("titleType") == "tvSeries")

        episode_count_df = episode_df.groupBy("parentTconst").agg(count("tconst").alias("numEpisodes"))

        result_df = episode_count_df.join(
            tv_series_df.select("tconst", "primaryTitle"),
            episode_count_df["parentTconst"] == tv_series_df["tconst"],
            how="inner"
        ).select("primaryTitle", "numEpisodes").orderBy(col("numEpisodes").desc()).limit(20)

        write_to_csv(result_df, self.output_path, "top_longest_tv_series_by_episodes")

        return result_df

    def run_all_analytics(self):
        self.most_mentioned_titles_by_actors().show(20, truncate=False)
        self.avg_episodes_per_season_over_time().show(20, truncate=False)
        self.average_gap_between_director_movies().show(20, truncate=False)
        self.avg_votes_per_year_high_rated().show(20, truncate=False)
        self.count_identical_titles().show(20, truncate=False)
        self.top_longest_tv_series_by_episodes().show(20, truncate=False)
