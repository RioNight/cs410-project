"""
Honkai: Star Rail Review Scraper
Collects reviews from Google Play Store using google-play-scraper library

Usage:
    python collect_reviews.py

Output:
    reviews.csv - CSV file with all collected reviews
"""

from google_play_scraper import app, Sort, reviews_all
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Configuration
APP_ID = 'com.HoYoverse.hkrpgoversea'  # Honkai: Star Rail
LANGUAGE = 'en'
COUNTRY = 'us'

print("="*70)
print("Honkai: Star Rail Review Scraper")
print("="*70)
print(f"\nApp ID: {APP_ID}")
print(f"Language: {LANGUAGE}")
print(f"Country: {COUNTRY}")
print(f"\nStarting scrape... (this may take several minutes)")
print("-"*70)

try:
    # Fetch all reviews
    g_reviews = reviews_all(
        APP_ID,
        sleep_milliseconds=0,  # No delay between requests
        lang=LANGUAGE,
        country=COUNTRY,
        sort=Sort.NEWEST,  # Get newest reviews first
    )

    print(f"\nSuccessfully fetched {len(g_reviews):,} reviews!")

    # Convert to DataFrame
    g_df = pd.DataFrame(np.array(g_reviews), columns=['review'])
    g_df2 = g_df.join(pd.DataFrame(g_df.pop('review').tolist()))

    # Clean up columns
    g_df2.drop(columns={'userImage', 'reviewCreatedVersion'}, inplace=True, errors='ignore')

    # Rename columns to match expected format
    g_df2.rename(columns={
        'score': 'rating',
        'userName': 'user_name',
        'reviewId': 'review_id',
        'content': 'review_description',
        'at': 'review_date',
        'replyContent': 'developer_response',
        'repliedAt': 'developer_response_date',
        'thumbsUpCount': 'thumbs_up'
    }, inplace=True)

    # Add metadata columns
    g_df2.insert(loc=0, column='source', value='Google Play')
    g_df2.insert(loc=3, column='review_title', value=None)
    g_df2['laguage_code'] = LANGUAGE
    g_df2['country_code'] = COUNTRY

    # Save to CSV
    output_file = 'reviews.csv'
    g_df2.to_csv(output_file, index=False)

    print(f"\nData saved to: {output_file}")
    print("\nDataset Summary:")
    print("-"*70)
    print(f"Total reviews: {len(g_df2):,}")
    print(f"Date range: {g_df2['review_date'].min()} to {g_df2['review_date'].max()}")
    print(f"Average rating: {g_df2['rating'].mean():.2f}/5.0")
    print(f"Reviews with developer response: {g_df2['developer_response'].notna().sum():,}")

    print("\nColumn names:")
    for col in g_df2.columns:
        print(f"  - {col}")

    print("\n" + "="*70)
    print("Scraping complete! You can now run the sentiment analysis notebook.")
    print("="*70)

except Exception as e:
    print(f"\nERROR: Failed to scrape reviews")
    print(f"Error message: {str(e)}")
    print("\nPossible solutions:")
    print("1. Check your internet connection")
    print("2. Verify the app ID is correct")
    print("3. Install required library: pip install google-play-scraper")
    print("4. Try again later (rate limiting)")
