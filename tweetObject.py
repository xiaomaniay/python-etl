import mysql.connector
from mysql.connector import Error
import os
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob

# nltk.download()

class TweetObject:

    def __init__(self, host, database, user):
        self.password = "MyNewPass"
        self.host = host
        self.database = database
        self.user = user

    def MySQLConnect(self, query):
        """
        Connects to database and extracts raw tweets
        """

        try:
            con = mysql.connector.connect(host=self.host, database=self.database, user=self.user, password=self.password, charset="utf8")

            if con.is_connected():
                print("Successfully connected to database")

                cursor = con.cursor()
                query = query
                cursor.execute(query)

                data = cursor.fetchall()

                # store fetched data into dataframe
                df = pd.DataFrame(data, columns=['date', 'tweet'])

                cursor.close()

            con.close()

        except Error as e:
            print(e)

        return df

    def clean_tweets(self, df):
        """
        Takes raw tweets and cleans them
        so we can carry out analysis
        remove stopwords, punctuation,
        lower case, html, emoticons.
        This will be done using Regex
        ? means option so colou?r matches
        both color and colour.
        """

        # text preprocessing
        stopword_list = stopwords.words('english')
        wordnet_lemmatizer = WordNetLemmatizer()
        df["clean_tweets"] = None
        df['len'] = None
        for i in range(0, len(df['tweet'])):
            # get rid of anything that isnt a letter

            exclusion_list = ['[^a-zA-Z]', 'rt', 'http', 'co', 'RT']
            exclusions = '|'.join(exclusion_list)
            text = re.sub(exclusions, ' ', df['tweet'][i])
            text = text.lower()
            words = text.split()
            words = [wordnet_lemmatizer.lemmatize(word) for word in words if word not in stopword_list]

            df['clean_tweets'][i] = ' '.join(words)

        # Create column with data length
        df['len'] = np.array([len(tweet) for tweet in df["clean_tweets"]])
        print(df.head())

        return df

    def sentiment(self, tweet):
        """
        This function calculates sentiment
        from our base on our cleaned tweets.
        Uses textblob to calculate polarity.
        Parameters:
        ----------------
        arg1: takes in a tweet (row of dataframe)
        ----------------
        Returns:
            Sentiment:
            1 is Positive
            0 is Neutral
               -1 is Negative
        """

        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def save_to_csv(self, df):
        """
        Save cleaned data to a csv for further
        analysis.
        Parameters:
        ----------------
        arg1: Pandas dataframe
        """
        try:
            df.to_csv("clean_tweets.csv")
            print("\n")
            print("csv successfully saved. \n")

        except Error as e:
            print(e)

    def word_cloud(self, df):
        """
        Takes in dataframe and plots a wordcloud using matplotlib
        """
        plt.subplots(figsize=(12, 10))
        wordcloud = WordCloud(
            background_color='white',
            width=1000,
            height=800).generate(" ".join(df['clean_tweets']))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()
