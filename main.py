from tweetObject import TweetObject
import numpy as np

t = TweetObject(host='localhost', database='twitterdb', user='root')

data = t.MySQLConnect("SELECT created_at, tweet FROM `twitterDB`.`golf`;")
data = t.clean_tweets(data)
data['Sentiment'] = np.array([t.sentiment(x) for x in data['clean_tweets']])
t.word_cloud(data)
t.save_to_csv(data)

pos_tweets = [tweet for index, tweet in enumerate(data["clean_tweets"]) if data["Sentiment"][index] > 0]
neg_tweets = [tweet for index, tweet in enumerate(data["clean_tweets"]) if data["Sentiment"][index] < 0]
neu_tweets = [tweet for index, tweet in enumerate(data["clean_tweets"]) if data["Sentiment"][index] == 0]

# Print results
print("percentage of positive tweets: {}%".format(100 * (len(pos_tweets) / len(data['clean_tweets']))))
print("percentage of negative tweets: {}%".format(100 * (len(neg_tweets) / len(data['clean_tweets']))))
print("percentage of neutral tweets: {}%".format(100 * (len(neu_tweets) / len(data['clean_tweets']))))
