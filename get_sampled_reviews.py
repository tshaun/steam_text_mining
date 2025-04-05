import os
import pandas as pd

# List of games
games = [
    # RPG
    "Dota 2",
    "The Elder Scrolls V: Skyrim",
    "The Witcher 3: Wild Hunt",

    # FPS
    "Call of Duty: Modern Warfare 3",
    "Counter-Strike",
    "DOOM",

    # Sports
    "NBA 2K16",
    "Rocket League",
    "Football Manager 2016"
]

# Path to sentiment.csv
sentiment_path = 'steam_reviews.csv'

# Check if the file exists
if os.path.exists(sentiment_path):
    # Load the sentiment.csv file
    sentiment_df = pd.read_csv(sentiment_path)

    # Filter and sample 25 reviews for each game
    sampled_reviews = sentiment_df[sentiment_df['app_name'].isin(games)].groupby('app_name').apply(
        lambda x: x.sample(n=25, replace=True) if len(x) >= 25 else x
    )

    # Reset index for the final DataFrame
    sampled_reviews.reset_index(drop=True, inplace=True)

    # Save the result to a new CSV file
    output_path = 'sampled_reviews.csv'
    sampled_reviews.to_csv(output_path, index=False)

    print(f"Sampled reviews saved to {output_path}")
else:
    print("File not found: processed_data/sentiment.csv")
