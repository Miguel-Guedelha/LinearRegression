import random
import os
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd
from lxml import etree
import re
from emoji import UNICODE_EMOJI
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
np.random.seed(0)

# Reading in English meta data (spreader or not)
def preprocess(r, pathlist):
    data = r.read().split("\n")
    idk = [] #id
    spreader = [] #yes or no
    for line in data:
        l = line.split(":::")
        if len(l)>1:
            idk.append(l[0])
            spreader.append(l[1])

    meta_data=pd.DataFrame()
    meta_data["ID"]=idk
    meta_data["spreader"]=spreader

    # Reading in and concatenating English tweets
 
    ids=[]
    x_raw=[]

    for path in pathlist:
        #iterate files
        head, tail = os.path.split(path)
        t=tail.split(".")
        author=t[0]
        ids.append(author)
        path_in_str = str(path)
        tree = ET.parse(path_in_str)
        root = tree.getroot()

        for child in root:
            xi=[]
            for ch in child:
                xi.append(ch.text)
            content = ' '.join(xi)
            x_raw.append(content)
    text_data=pd.DataFrame()
    text_data["ID"]=ids
    text_data["Tweets"]=x_raw

    # Merging meta data and text data to one dataframe
    data = pd.merge(meta_data, text_data, how='inner', on = 'ID')
    feed_list = data["Tweets"].tolist()

    return feed_list, data

def intiial_cleaning(tweet_lista):
    intiial_cleaning=[]
    for feed in tweet_lista:
        feed = feed.lower()
        feed = re.sub('[^0-9a-z #@]', "", feed)
        feed = re.sub('[\n]', " ", feed)
        intiial_cleaning.append(feed)
       
    return intiial_cleaning


def hashtag(tweet_lista):
    hashtag=[]
    for feed in tweet_lista:
        feed = feed.lower()
        feed = re.sub('[^0-9a-z #@]', "", feed)
        feed = re.sub('[\n]', " ", feed)
        feed = re.findall(r'#\S+', feed)
        hashtag.append(feed)
    hashtag = [' '.join(i) for i in  hashtag]
    return hashtag

def userMention(tweet_lista):
    userMention=[]
    for feed in tweet_lista:
        feed = feed.lower()
        feed = re.sub('[^0-9a-z #@]', "", feed)
        feed = re.sub('[\n]', " ", feed)
        feed = re.findall(r'@\S+', feed)
        userMention.append(feed)
    userMention = [' '.join(i) for i in  userMention]
  
    return userMention


def vectorization(cleanedTweets, data, p1, p2):

    tfidf = TfidfVectorizer(ngram_range = (p1,p2), use_idf=True, smooth_idf=True, sublinear_tf=True)
    x = tfidf.fit_transform(cleanedTweets)
    y = data["spreader"]
   
    return x, y, tfidf

def regres(cleanedTweets, data):
    x, y, tfidf = vectorization(cleanedTweets, data, 1, 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.5, shuffle=True)
    lr = LogisticRegression(solver='liblinear',verbose=0, random_state=5).fit(X_train, y_train)
    return X_test, y_test, lr, tfidf

def predictions(X_test,y_test, filename):
    saved_lr = pickle.load(open(filename, 'rb'))
    a = saved_lr.score(X_test, y_test)
    prediction = saved_lr.predict(X_test)
    cv = precision_recall_fscore_support(y_test, prediction, average='macro')
    print(a)
    return a

    
def plot1(accuracy, model):
    plt.title('Predicting COVID-19 fake news super spreaders (n-grams)', fontsize=18)
    plt.xlabel('model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.bar(model[0], accuracy[0])
    plt.bar(model[1], accuracy[1])
    plt.bar(model[2], accuracy[2])
    plt.legend()
    plt.show()

    plt.title('Predicting COVID-19 fake news super spreaders (hashtags)', fontsize=18)
    plt.xlabel('model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.bar(model[3], accuracy[3])
    plt.bar(model[4], accuracy[4])
  
    plt.legend()
    plt.show()

    plt.title('Predicting COVID-19 fake news super spreaders (VS real news spreaders)', fontsize=18)
    plt.xlabel('model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.bar(model[5], accuracy[5])
    plt.bar(model[6], accuracy[6])
    plt.bar(model[7], accuracy[7])
    plt.legend()
    plt.show()
    return

# ngram , fake vs not fake, n gram
print('fake vs not fake')

r = open('/Users/chemuitrickard/FakeNewsMCS/fake news datasets/datasets (1)/fake-vs-non_fake-unfiltered copy/FakeNewsFinal/combinedFakeVSNotFaketruth.txt', "r")
pathlist = Path('/Users/chemuitrickard/FakeNewsMCS/fake news datasets/datasets (1)/fake-vs-non_fake-unfiltered copy/FakeNewsFinal/combinedFILES').glob('**/*.xml')
feed_List, data = preprocess(r, pathlist)
cleanedTweets = intiial_cleaning(feed_List)
x, y, lr, fVector = regres(cleanedTweets, data)
fake_news_model = open('fake_news_model.sav', 'wb')
pickle.dump(lr, fake_news_model)
fake_news_model.close()

# fake vs not fake, hashtag 
print('covid fake news spreaders vs people who have not spread fake news, # analysis')
r = open('/Users/chemuitrickard/FakeNewsMCS/fake news datasets/datasets (1)/fake-vs-non_fake-unfiltered copy/FakeNewsFinal/combinedFakeVSNotFaketruth.txt', "r")
pathlist = Path('/Users/chemuitrickard/FakeNewsMCS/fake news datasets/datasets (1)/fake-vs-non_fake-unfiltered copy/FakeNewsFinal/combinedFILES').glob('**/*.xml')
feed_List, data = preprocess(r, pathlist)
cleanedTweets = hashtag(feed_List)
xH, yH, lr, hVector = regres(cleanedTweets, data)
fake_news_model_hashtag = open('fake_news_model_hashtag.sav', 'wb')
pickle.dump(lr, fake_news_model_hashtag)
fake_news_model_hashtag.close()

# ngram , fake vs real, n gram
print('fake vs not fake')
r = open('/Users/chemuitrickard/FakeNewsMCS/fake news datasets/datasets (1)/fake-vs-non_fake-unfiltered copy/FakeNewsFinal/combinedFakeVSReal.txt', "r")
pathlist = Path('/Users/chemuitrickard/FakeNewsMCS/fake news datasets/datasets (1)/fake-vs-non_fake-unfiltered copy/FakeNewsFinal/combinedFakeVSRealFILES').glob('**/*.xml')
feed_List, data = preprocess(r, pathlist)
cleanedTweets = intiial_cleaning(feed_List)
xR, yR, lr, RVector = regres(cleanedTweets, data)
fake_news_model_real = open('fake_news_model_real.sav', 'wb')
pickle.dump(lr, fake_news_model_real)
fake_news_model_real.close()




# fake vs real, n gram
# user mentions, fake vs not fake
r = open('/Users/chemuitrickard/FakeNewsMCS/fake news datasets/datasets (1)/fake-vs-non_fake-unfiltered copy/FakeNewsFinal/combinedFakeVSNotFaketruth.txt', "r")
pathlist = Path('/Users/chemuitrickard/FakeNewsMCS/fake news datasets/datasets (1)/fake-vs-non_fake-unfiltered copy/FakeNewsFinal/combinedFILES').glob('**/*.xml')
pathlist = Path('/Users/chemuitrickard/chemFakeNews/hashtag/files').glob('**/*.xml')
feed_List, data = preprocess(r, pathlist)
cleanedTweets = userMention(feed_List)
xA, yA, lr, uVector= regres(cleanedTweets, data)
fake_news_model_mentions = open('fake_news_model_mentions.sav', 'wb')
pickle.dump(lr, fake_news_model_mentions)
fake_news_model_mentions.close()









# predictions 
LRaccuracy = predictions(x,y, 'fake_news_model.sav')
hashtagAccuracy = predictions(xH,yH, 'fake_news_model_hashtag.sav')
RealLRAccuracy = predictions(xR,yR, 'fake_news_model_real.sav')


accuracy = []
model = ['LR', 'CNN', 'LSTM', 'LR - #', 'CNN - #', 'LR', 'CNN', 'LSTM']
accuracy.append(LRaccuracy)
# results from the CNN and LSTM
accuracy.append(0.95)
accuracy.append(0.82)
accuracy.append(hashtagAccuracy)
accuracy.append(0.80)
accuracy.append(RealLRAccuracy)
accuracy.append(0.95)
accuracy.append(0.82)
plot1(accuracy, model)



# this is for companies to imput their own twitter profiles and make predictions using the saved models

def newDataPredictions(cleanTweets, modelname):
    saved_lr = pickle.load(open(modelname, 'rb'))
    prediction = saved_lr.predict(cleanTweets)
    print(prediction)
    return

def userPreProcessing(pathForProfiles):
    pathlist = pathForProfiles
    newData=[]
    for path in pathlist:
        path_in_str = str(path)
        tree = ET.parse(path_in_str)
        root = tree.getroot()
        for child in root:
            xi=[]
            for ch in child:
                xi.append(ch.text)
            content = ' '.join(xi)
            newData.append(content)
    return newData

newData = userPreProcessing(Path('/Users/chemuitrickard/chemFakeNews/hashtag/files').glob('**/*.xml'))
cleanedTweets = intiial_cleaning(newData)
x = fVector.transform(cleanedTweets)
LRaccuracy = newDataPredictions(x, 'fake_news_model.sav')
Path('/Users/chemuitrickard/chemFakeNews/hashtag/files').glob('**/*.xml')
