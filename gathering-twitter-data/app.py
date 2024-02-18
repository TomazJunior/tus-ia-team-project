import tweepy
import sys
import datetime
import csv
import json

with open('./config.json', 'r') as file:
    config = json.load(file)

consumer_key = config['twitter']['consumer_key']
consumer_secret = config['twitter']['consumer_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# Twitter authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")
    sys.exit()

# Define the search term and date range
search_term = 'Bitcoin'
start_date = datetime.datetime(2022, 1, 1, 0, 0, 0)
end_date = datetime.datetime(2022, 12, 31, 23, 59, 59)

with open('./data/tweets.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # CSV Header
    writer.writerow(['id', 'created_at', 'name', 'text'])

    # Fetch 10 tweets for now
    for tweet in tweepy.Cursor(api.search_tweets,
                           q=search_term,
                           lang="en",
                           since=start_date.strftime('%Y-%m-%d'),
                           until=end_date.strftime('%Y-%m-%d')).items(2):
        print(f"{tweet.id} - {tweet.created_at} - {tweet.user.name} - {tweet.text}")
        writer.writerow([tweet.id, tweet.created_at, tweet.user.name, tweet.text])
        
