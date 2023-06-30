import pybaseball

pybaseball.cache.enable()

# Just get any data.
# some_pitches = pybaseball.statcast(start_dt="2019-01-01", end_dt="2022-12-31")

# some_pitches.to_parquet("data/bigger_training_data.parquet")

# Just get any data.
some_more_pitches = pybaseball.statcast(start_dt="2023-01-01", end_dt="2023-12-31")

some_more_pitches.to_parquet("data/2023_data.parquet")
