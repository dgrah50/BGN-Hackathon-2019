from __future__ import division
from twitterscraper import query_tweets
import sys, csv, re, string, heapq, gensim


from collections import Counter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from gensim import corpora, models
# from textblob import TextBlob
#






def fetchTweetsFromFile(twitter_data):
    fields = []
    rows = []

    with open(twitter_data, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')

        # This skips the first row of the CSV file.
        # https://evanhahn.com/python-skip-header-csv-reader/
        next(csvreader)

        for row in csvreader:
            rows.append(row)
        print
        "Total no. of tweets: {}".format(csvreader.line_num - 2)

    return rows


def clean_text_and_tokenize(line):
    from nltk.corpus import stopwords
    from nltk.tokenize import TweetTokenizer
    from nltk.stem.wordnet import WordNetLemmatizer
    lemma = WordNetLemmatizer()
    sw = stopwords.words('english')
    line = re.sub(r'\$\w*', '', line)  # Remove tickers
    line = re.sub(r'http?:.*$', '', line)
    line = re.sub(r'https?:.*$', '', line)
    line = re.sub(r'pic?.*\/\w*', '', line)
    line = re.sub(r'[' + string.punctuation + ']+', ' ', line)  # Remove puncutations like 's

    tokens = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(line)
    tokens = [w.lower() for w in tokens if w not in sw and len(w) > 2 and w.isalpha()]
    tokens = [lemma.lemmatize(word) for word in tokens]

    return tokens


def getCleanedWords(lines):
    words = []

    for line in lines:
        words += clean_text_and_tokenize(line)
    return words


def lexical_diversity(tokens):
    return 1.0 * len(set(tokens)) / len(tokens)


# Helper function for computing number of words per tweet
def average_words(lines):
    total_words = sum([len(s.split()) for s in lines])

    return 1.0 * total_words / len(lines)


def top_words(words, top=5):
    c = Counter(words)
    # print(words)
    return (c)


def popular_tweets(tweet_rows, top=5):
    popular = []
    for row in tweet_rows:
        if len(row) >= 8:
            popular.append([row[8], int(row[2]) + int(row[3]), row[4], row[9]])
    topTweets = heapq.nlargest(
        top, popular, key=lambda e: e[1])  # ref sof -> 2243542

    print("\nPrinting top {} tweets".format(top))
    counter = 0
    for (id, popularity, tweet, url) in topTweets:
        counter += 1
        print("{}. {}".format(counter, tweet))
        print("Popularity = {}".format(popularity))
        print("Link = {}".format(url))
        print("-------------------")


def clean_tweet(tweet):
    return " ".join(clean_text_and_tokenize(tweet))


def sentiment_analysis_basic(tweets,latitude,longitude):
    from textblob import TextBlob
    positive_tweets = 0
    neutral_tweets = 0
    negative_tweets = 0

    for tweet in tweets:
        analysis = TextBlob(tweet)
        sentiment = analysis.sentiment.polarity

        if sentiment > 0:
            positive_tweets += 1
        elif sentiment == 0:
            neutral_tweets += 1
        else:
            negative_tweets += 1
    total_tweets_analysed = positive_tweets + neutral_tweets + negative_tweets
    positive_tweets_percentage = positive_tweets / total_tweets_analysed * 100
    neutral_tweets_percentage = neutral_tweets / total_tweets_analysed * 100
    negative_tweets_percentage = negative_tweets / total_tweets_analysed * 100
    output = {}
    output['happysent'] = positive_tweets_percentage
    output['neutralsent'] = neutral_tweets_percentage
    output['negativesent'] = negative_tweets_percentage
    output['latitude'] = latitude
    output['longitude'] = longitude
    # return (("happysent:",positive_tweets_percentage),("neutralsent:", neutral_tweets_percentage),("negativesent:",negative_tweets_percentage))
    return output
    # ("\nNo. of positive tweets = {} Percentage = {}".format(positive_tweets, positive_tweets_percentage))

    # print("No. of neutral tweets  = {} Percentage = {}".format(
    #     neutral_tweets, neutral_tweets_percentage))
    # print(
    # "No. of negative tweets = {} Percentage = {}".format(
    #     negative_tweets, 100 - (positive_tweets_percentage + neutral_tweets_percentage)))


# sof -> 20078816
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

#
# def clusterTweetsKmeans(tweets):
#     taggeddocs = []
#     tag2tweetmap = {}
#     for index, i in enumerate(tweets):
#         if len(i) > 2:  # Non empty tweets
#             tag = u'SENT_{:d}'.format(index)
#             sentence = TaggedDocument(
#                 words=gensim.utils.to_unicode(i).split(), tags=[tag])
#             tag2tweetmap[tag] = i
#             taggeddocs.append(sentence)
#
#     model = Doc2Vec(
#         taggeddocs, dm=0, alpha=0.025, size=20, min_alpha=0.025, min_count=0)
#     print(" ")
#     for epoch in range(60):
#         model.train(
#             taggeddocs, total_examples=model.corpus_count, epochs=model.iter)
#         model.alpha -= 0.002  # decrease the learning rate
#         model.min_alpha = model.alpha  # fix the learning rate, no decay
#
#     dataSet = model.docvecs.doctag_syn0  # this works, thanks a lot sof -> 43476869
#     kmeansClustering = KMeans(n_clusters=5)
#     centroidIndx = kmeansClustering.fit_predict(dataSet)
#     topic2wordsmap = {}
#     for i, val in enumerate(dataSet):
#         tag = model.docvecs.index_to_doctag(i)
#         topic = centroidIndx[i]
#         if topic in topic2wordsmap.keys():
#             for w in (tag2tweetmap[tag].split()):
#                 topic2wordsmap[topic].append(w)
#         else:
#             topic2wordsmap[topic] = []
#     for i in topic2wordsmap:
#         print(
#             "Topic {} has words: {}".format(i + 1,
#                                             ' '.join(remove_non_ascii(word) for word in topic2wordsmap[i][:10])))


def analysis(tweetbylocation):

    from nltk.tokenize import TweetTokenizer
    from nltk.stem.wordnet import WordNetLemmatizer


    lemma = WordNetLemmatizer()
    # twitter_data = "/tmp/data.csv"
    # tweet_rows = fetchTweetsFromFile(twitter_data)

    # for tweet in tweet_rows:
    #     print(tweet)
    # tweets = [row[0] for row in tweet_rows]
    locstats={}
    for loc in tweetbylocation :
        # print (str(loc[0][0]) + "---" + str(loc[0][1]) + '----' + str(loc[0][2]))
        tweets=loc[1]

        finalstats = []
        # finalstats.append(average_words(tweets))

        # print ("Average Number of words per tweet = {}".format(average_words(tweets)))
        words = getCleanedWords(tweets)
        # print("Lexical diversity = {}".format(lexical_diversity(words)))
        # finalstats.append(lexical_diversity(words))
        # print (words)
        # print ("Top Words = {}".format(top_words(words)))
        # print(top_words(words))
        # finalstats.append(top_words(words))
        # popular_tweets(tweets)
        # print(finalstats)
        cleaned_tweets = []
        for tweet in tweets:
            cleaned_tweets.append(clean_tweet(tweet))
        # finalstats.append(sentiment_analysis_basic(cleaned_tweets))
        # print("SENTIMENT SHOULD BE HERE")
        # print(sentiment_analysis_basic(cleaned_tweets))
        # clusterTweetsKmeans(clean§ed_tweets)
        locstats[loc[0][0]] = sentiment_analysis_basic(cleaned_tweets,loc[0][1],loc[0][2])
        # locstats.update((loc[0],sentiment_analysis_basic(cleaned_tweets)))
        # print(locstats)
    return locstats



def fetch(query):

    locations = [("Shoreditch",51.529440,-0.077540), ("Greenwich",51.478790,-0.010680), ("Brixton",51.461281,-0.115615),
                 ("Camden",51.539190,-0.142500), ("Paddington",51.515970,-0.174970), ("Tooting",51.438240,-0.156670), ("Stratford",51.538735,0.001300)]
    # print the retrieved tweets to the screen:
    tweetsbylocation = []
    for location in locations:
        # print("************" + location + "************")
        print(query + " near:'" + location[0] )
        tweets = []
        for tweet in query_tweets((query + " near:" + location[0] ), 10):
            # print (tweet.text)
            tweets.append(tweet.text)
        # print(analysis(tweets))
        tweetsbylocation.append((location,tweets))
    return (analysis(tweetsbylocation))


    # for loc in tweetsbylocation:
    #     print("***************"+loc[0]+"***************")
    #
    #     print(analysis(loc[1]))
    #     # print(type(loc))

if __name__ == '__main__':
    main()