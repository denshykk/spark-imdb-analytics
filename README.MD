# Spark IMDB Analytics

## Overview
A comprehensive analysis of the IMDB dataset using Apache Spark, focusing on extracting meaningful insights through SQL queries, data transformation, and analytical processing.

## Project Description
This project is part of the "Big Data Analysis" course, aiming to develop skills in working with Apache Spark for distributed data processing. We analyze the IMDB dataset to answer business questions using various SQL techniques including filtering, joins, group by operations, and window functions.

## Dataset
We use the official IMDB dataset available at [datasets.imdbws.com](https://datasets.imdbws.com/), which includes:

- **name.basics.tsv.gz**: Information about people (directors, actors, writers)
- **title.akas.tsv.gz**: Alternative titles across different regions
- **title.basics.tsv.gz**: Basic information about titles (movies, TV shows, etc.)
- **title.crew.tsv.gz**: Directors and writers for each title
- **title.episode.tsv.gz**: Episode information for TV series
- **title.principals.tsv.gz**: Principal cast/crew for titles
- **title.ratings.tsv.gz**: IMDb ratings for titles

## Installation

### Prerequisites
- Python 3.8 or higher
- Java 8 or 11 (required for Spark)
- Docker

### Setup Instructions

#### Option 1: Direct Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/denshykk/spark-imdb-analytics.git
   cd spark-imdb-analytics
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the IMDB dataset:
   ```bash
   ./scripts/download-imdb-dataset.sh
   ```
   
5. Run the main application:
```bash
python main.py
```

#### Option 2: Docker
1. Clone the repository:
   ```bash
   git clone https://github.com/denshykk/spark-imdb-analytics.git
   cd spark-imdb-analytics
   ```

2. Build the Docker image:
   ```bash
   docker build -t my-spark-img .
   ```

3. Run the application:
   ```bash
   docker run my-spark-img
   ```

## Project Structure
```
spark-imdb-analytics/
├── Dockerfile
├── README.MD
├── main.py
├── requirements.txt
└── scripts
    └── download-imdb-dataset.sh
```

## Business Questions
Our analysis answers the following business questions:

1. TBD - We will develop 6 business questions covering:
   - Filtering operations (at least 3 questions)
   - Join operations (at least 2 questions)
   - Group by operations (at least 2 questions)
   - Window functions (at least 2 questions)

## Team
This project is being developed by:

- Denys Tykhonov
- Ivan Dobrodieiev
- Ivan Pivtorak
- Marta Oliinyk

## License
This project is for educational purposes. The IMDB dataset is subject to IMDb's terms of use for non-commercial datasets.
