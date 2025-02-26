#!/bin/bash

mkdir -p ../imdb_data
cd ../imdb_data

echo "Downloading IMDB datasets..."

echo "Downloading name.basics.tsv.gz..."
curl -O https://datasets.imdbws.com/name.basics.tsv.gz

echo "Downloading title.akas.tsv.gz..."
curl -O https://datasets.imdbws.com/title.akas.tsv.gz

echo "Downloading title.basics.tsv.gz..."
curl -O https://datasets.imdbws.com/title.basics.tsv.gz

echo "Downloading title.crew.tsv.gz..."
curl -O https://datasets.imdbws.com/title.crew.tsv.gz

echo "Downloading title.episode.tsv.gz..."
curl -O https://datasets.imdbws.com/title.episode.tsv.gz

echo "Downloading title.principals.tsv.gz..."
curl -O https://datasets.imdbws.com/title.principals.tsv.gz

echo "Downloading title.ratings.tsv.gz..."
curl -O https://datasets.imdbws.com/title.ratings.tsv.gz

echo "All downloads complete!"
ls -lh
