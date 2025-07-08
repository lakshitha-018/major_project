import streamlit as st
import tweepy
import pandas as pd
import time
import random
import traceback
from transformers import pipeline

# --------- Twitter API Setup ---------
bearer_token = "AAAAAAAAAAAAAAAAAAAAAIoT1AEAAAAAOEjMmPeqKJ6nkc419erghX%2B45mY%3DCBjh0qxwAq5M6YMNEDOBV5uS7ECC1PzSqSu7j0cifsp19OtxD2"  # Replace with actual decoded token
client = tweepy.Client(bearer_token=bearer_token)

# --------- Load Sentiment Model ---------
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

pipe = load_model()

# --------- Sentiment Mapping ---------
def map_sentiment(label):
    label = label.lower()
    if "very negative" in label or "negative" in label:
        return "negative"
    elif "neutral" in label:
        return "neutral"
    elif "positive" in label or "very positive" in label:
        return "positive"
    else:
        return "unknown"

# --------- Offline Fake Tweets ---------
def generate_fake_tweets(keyword, count):
    templates = [
        f"I really love how {keyword} is changing the world!",
        f"Wow! {keyword} blew my mind today. Amazing stuff!",
        f"{keyword} just keeps getting better and better.",
        f"The best thing I saw today? {keyword}, no doubt.",
        f"{keyword} is the future of technology.",
        f"{keyword} is overrated. Totally disappointed.",
        f"I can't believe how terrible {keyword} has become.",
        f"Why is everyone talking about {keyword}? It's boring.",
        f"{keyword} was mentioned in the news today.",
        f"Just heard about {keyword}.",
        f"{keyword} exists, and that’s fine.",
        f"Some people are talking about {keyword}.",
        f"{keyword} trends again.",
        f"{keyword} was discussed in today’s seminar.",
        f"No strong opinions about {keyword}, just interesting.",
        f"{keyword} is a topic I heard recently.",
    ]
    return random.choices(templates, k=count)

# --------- Paginated Twitter Search ---------
def safe_search(query, count, max_retries=3):
    all_tweets = []
    next_token = None
    retries = 0

    while len(all_tweets) < count and retries < max_retries:
        try:
            response = client.search_recent_tweets(
                query=query + " -is:retweet lang:en",
                max_results=min(100, count - len(all_tweets)),
                tweet_fields=["text"],
                next_token=next_token
            )
            if response.data:
                all_tweets.extend(response.data)
                next_token = response.meta.get("next_token")
                if not next_token:
                    break  # No more pages
            else:
                break  # No more tweets

        except tweepy.TooManyRequests:
            retries += 1
            wait_time = 90
            st.warning(f"⏳ Rate limit reached. Retry {retries}/{max_retries}... Waiting {wait_time} seconds.")
            time.sleep(wait_time)
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.text(traceback.format_exc())
            return None

    if retries == max_retries:
        st.error("❌ Max retries reached. Try again later or use Offline Mode.")
        return None

    st.info(f"✅ Fetched {len(all_tweets)} tweets.")
    return all_tweets

# --------- UI ---------
st.title("🌍 Real-Time Twitter Sentiment Analyzer")

query = st.text_input("Enter keyword or hashtag to track", value="technology")
tweet_count = st.slider("Select the number of tweets to analyze", 1, 500, 50)  # Up to 500
offline_mode = st.checkbox("🔌Use API")

# --------- Run Button ---------
if st.button("Run Sentiment Analysis") and query:
    with st.spinner("Analyzing tweets..."):
        tweet_data = []

        try:
            if offline_mode:
                fake_tweets = generate_fake_tweets(query, tweet_count)
                for text in fake_tweets:
                    sentiment = pipe(text)[0]
                    tweet_data.append({
                        "text": text,
                        "sentiment": map_sentiment(sentiment["label"]),
                        "confidence": round(sentiment["score"], 3)
                    })
            else:
                response = safe_search(query, tweet_count)
                if response:
                    for tweet in response:
                        sentiment = pipe(tweet.text)[0]
                        tweet_data.append({
                            "text": tweet.text,
                            "sentiment": map_sentiment(sentiment["label"]),
                            "confidence": round(sentiment["score"], 3)
                        })
                else:
                    st.warning("No tweets found or request failed.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.text(traceback.format_exc())
            st.stop()

        # --------- Show Results ---------
        if tweet_data:
            tweet_df = pd.DataFrame(tweet_data)

            st.subheader("📊 Analyzed Tweets")
            st.dataframe(tweet_df)

            st.subheader("📈 Sentiment Distribution")
            sentiment_counts = tweet_df["sentiment"].value_counts()
            sentiment_percentages = (sentiment_counts / len(tweet_df)) * 100

            for sentiment, percentage in sentiment_percentages.items():
                st.write(f"{sentiment.capitalize()}: {percentage:.2f}%")

            st.bar_chart(sentiment_counts)
