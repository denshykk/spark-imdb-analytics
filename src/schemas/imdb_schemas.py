from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Schema for name.basics.tsv
# nconst (string) - alphanumeric unique identifier of the name/person
# primaryName (string) - name by which the person is most often credited
# birthYear (integer) - in YYYY format
# deathYear (integer) - in YYYY format if applicable, else '\N'
# primaryProfession (array of strings) - the top-3 professions of the person
# knownForTitles (array of string) - titles the person is known for
name_basics_schema = StructType([
    StructField("nconst", StringType(), False),
    StructField("primaryName", StringType(), True),
    StructField("birthYear", IntegerType(), True),
    StructField("deathYear", IntegerType(), True),
    StructField("primaryProfession", StringType(), True),  # Will parse as array later
    StructField("knownForTitles", StringType(), True),  # Will parse as array later
])

# Schema for title.akas.tsv
# titleId (string) - a tconst, an alphanumeric unique identifier of the title
# ordering (integer) - a number to uniquely identify rows for a given titleId
# title (string) - the localized title
# region (string) - the region for this version of the title
# language (string) - the language of the title
# types (array of strings) - Enumerated set of attributes for this alternative title
# attributes (array of strings) - Additional terms to describe this alternative title
# isOriginalTitle (boolean) - 0: not original title; 1: original title
title_akas_schema = StructType([
    StructField("titleId", StringType(), False),
    StructField("ordering", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("region", StringType(), True),
    StructField("language", StringType(), True),
    StructField("types", StringType(), True),  # Will parse as array later
    StructField("attributes", StringType(), True),  # Will parse as array later
    StructField("isOriginalTitle", IntegerType(), True)
])

# Schema for title.basics.tsv
# tconst (string) - alphanumeric unique identifier of the title
# titleType (string) - the type/format of the title (e.g. movie, short, tvseries, etc)
# primaryTitle (string) - the more popular title / the title used by the filmmakers on promotional materials at the point of release
# originalTitle (string) - original title, in the original language
# isAdult (boolean) - 0: non-adult title; 1: adult title
# startYear (integer) - represents the release year of a title. In the case of TV Series, it is the series start year
# endYear (integer) - TV Series end year. '\N' for all other title types
# runtimeMinutes (integer) - primary runtime of the title, in minutes
# genres (array of strings) - includes up to three genres associated with the title
title_basics_schema = StructType([
    StructField("tconst", StringType(), False),
    StructField("titleType", StringType(), True),
    StructField("primaryTitle", StringType(), True),
    StructField("originalTitle", StringType(), True),
    StructField("isAdult", IntegerType(), True),
    StructField("startYear", IntegerType(), True),
    StructField("endYear", IntegerType(), True),
    StructField("runtimeMinutes", IntegerType(), True),
    StructField("genres", StringType(), True),  # Will parse as array later
])

# Schema for title.crew.tsv
# tconst (string) - alphanumeric unique identifier of the title
# directors (array of strings) - director(s) of the given title
# writers (array of strings) - writer(s) of the given title
title_crew_schema = StructType([
    StructField("tconst", StringType(), False),
    StructField("directors", StringType(), True),  # Will parse as array later
    StructField("writers", StringType(), True),  # Will parse as array later
])

# Schema for title.episode.tsv
# tconst (string) - alphanumeric identifier of episode
# parentTconst (string) - alphanumeric identifier of the parent TV Series
# seasonNumber (integer) - season number the episode belongs to
# episodeNumber (integer) - episode number of the tconst in the TV series
title_episode_schema = StructType([
    StructField("tconst", StringType(), False),
    StructField("parentTconst", StringType(), True),
    StructField("seasonNumber", IntegerType(), True),
    StructField("episodeNumber", IntegerType(), True),
])

# Schema for title.principals.tsv
# tconst (string) - alphanumeric unique identifier of the title
# ordering (integer) - a number to uniquely identify rows for a given titleId
# nconst (string) - alphanumeric unique identifier of the name/person
# category (string) - the category of job that person was in
# job (string) - the specific job title if applicable, else '\N'
# characters (string) - the name of the character played if applicable, else '\N'
title_principals_schema = StructType([
    StructField("tconst", StringType(), False),
    StructField("ordering", IntegerType(), True),
    StructField("nconst", StringType(), True),
    StructField("category", StringType(), True),
    StructField("job", StringType(), True),
    StructField("characters", StringType(), True),  # JSON array as string, will need parsing
])

# Schema for title.ratings.tsv
# tconst (string) - alphanumeric unique identifier of the title
# averageRating (float) - weighted average of all the individual user ratings
# numVotes (integer) - number of votes the title has received
title_ratings_schema = StructType([
    StructField("tconst", StringType(), False),
    StructField("averageRating", FloatType(), True),
    StructField("numVotes", IntegerType(), True),
])

# Dictionary of all schemas
imdb_schemas = {
    "name.basics": name_basics_schema,
    "title.akas": title_akas_schema,
    "title.basics": title_basics_schema,
    "title.crew": title_crew_schema,
    "title.episode": title_episode_schema,
    "title.principals": title_principals_schema,
    "title.ratings": title_ratings_schema
}
