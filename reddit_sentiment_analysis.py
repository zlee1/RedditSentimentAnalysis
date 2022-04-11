import praw
from psaw import PushshiftAPI
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import traceback
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Suppress warning messages
import warnings

def daily_change(data):
    change = []
    up = []
    for index, row in data.iterrows():
        change.append(row["Close"]-row["Open"])
        up.append(int(change[-1] > 0))
    return change, up

def get_pre_open_content(data, ticker, start_hour_diff=0, subreddit="stocks,stockmarket,stocksandtrading,daytrading,investing,stocks_picks,stockstobuytoday", limit=100):
    new_col = []
    for index, row in data.iterrows():
        end_time = row.name + timedelta(hours=9, minutes=30)
        start_time = end_time - timedelta(hours=17, minutes=30)
        content = []
        for i in api.search_comments(after=start_time, before=end_time, subreddit=subreddit, q=ticker, filter=['url','author', 'title', 'subreddit'], limit=limit):
            for j in i.body.split("."):
                for k in j.split("\n"):
                    content.append(k)
        new_col.append(content)

    return new_col

def split_sequences(data):
    dates = []
    seqs = []
    vals = []

    for index, row in data.iterrows():
        for comment in row["Comment_Sequences"]:
            if(comment != []):
                dates.append(row.name)
                seqs.append(list(comment))
                vals.append(row["Positive_Change"])
    return pd.DataFrame({"Date":dates, "Sequence":seqs, "Positive_Change":vals})

def prepare_text(data, train_proportion = 0.8, max_len=50, tokenizer=None):
    if(tokenizer is None):
        tokenizer = Tokenizer(oov_token = "<OOV>")

    train = data[:int(data.shape[0]*train_proportion)]
    test = data[int(data.shape[0]*train_proportion):]

    for comment in train.Comments:
        tokenizer.fit_on_texts(comment)

    seqs = []
    for comment in train.Comments:
        seqs.append(tokenizer.texts_to_sequences(comment))

    train["Comment_Sequences"] = seqs

    train = split_sequences(train)

    train_padded = pad_sequences(train["Sequence"], padding="post", truncating="post", maxlen=max_len)

    for comment in test.Comments:
        tokenizer.fit_on_texts(comment)

    seqs = []
    for comment in test.Comments:
        seqs.append(tokenizer.texts_to_sequences(comment))

    test["Comment_Sequences"] = seqs

    test = split_sequences(test)

    test_padded = pad_sequences(test["Sequence"], padding="post", truncating="post", maxlen=max_len)

    return train, train_padded, test, test_padded, tokenizer

def generate_ticker_data(tickers = None):
    if(tickers is None): return None

    cur_tokenizer = None

    for ticker in tickers:
        if(f"{ticker}.pkl" in os.listdir('data/')):
            continue

        data = yf.download(ticker)
        change, up = daily_change(data)
        data["Daily_Change"] = change
        data["Positive_Change"] = up
        data = data.sample(frac=1)

        data["Comments"] = get_pre_open_content(data, ticker, limit=100)

        train, train_padded, test, test_padded, cur_tokenizer = prepare_text(data, tokenizer=cur_tokenizer)

        with open(f"data/{ticker}.pkl", "wb+") as f:
            pickle.dump((train, train_padded, test, test_padded), f)

        used_tickers = "_".join(tickers[:tickers.index(ticker)+1])

        with open(f"data/tokenizer/tokenizer_{used_tickers}.pkl", "wb+") as f:
            pickle.dump(tokenizer, f)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # Load client_id, secret_id, and user_agent
    with open('info.json') as f:
         info = json.load(f)

    info = dict(info)

    reddit = praw.Reddit(client_id=info["client_id"], user_agent=info["user_agent"], client_secret=info["client_secret"])
    api = PushshiftAPI(reddit)

    generate_ticker_data(["TWTR", "AMZN", "MSFT", "NCLH", "TSLA", "NFLX", "AAPL", "ADBE", "AMD", "DIS", "DAL", "SPOT", "SNAP", "GOOG", "JPM", "FB", "IBM", "T"])
