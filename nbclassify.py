import pandas as pd
import numpy as np
import re, nltk
import random
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
#nltk.download('punkt')
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems
########

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 85
)
train_data_f = "training.txt"
test_data_f = "reviews1.csv"

train_d = pd.read_csv(train_data_f, header=None ,delimiter="\t", quoting=3)
train_d.columns = ["Sentiment","Text"]

test_d = pd.read_csv(test_data_f, header=None, delimiter="\n", quoting=1)
test_d.columns = ["Text"]

#print test_d.head()
#print train_d.Sentiment.value_counts()

print train_d.shape

#np.mean([len(s.split(" ")) for s in train_d.Text])

corp_data_features = vectorizer.fit_transform(train_d.Text.tolist() + test_d.Text.tolist())

corp_data_features_nd = corp_data_features.toarray()
print corp_data_features_nd.shape
vocab = vectorizer.get_feature_names()

dist = np.sum(corp_data_features_nd, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the data set
#for tag,count in zip(vocab,dist):
 #   print count, tag

X_train, X_test, y_train,y_test = train_test_split(
    corp_data_features_nd[0:len(train_d)],
    train_d.Sentiment,
    train_size=0.85,
    random_state=1234)

#nb = GaussianNB()
#nb = nb.fit(X=X_train,y=y_train)

#y_pred = nb.predict(X_test)
#print(classification_report(y_test,y_pred))

nb = GaussianNB()
nb = nb.fit(X=corp_data_features_nd[0:len(train_d)],y=train_d.Sentiment)

test_pred = nb.predict(corp_data_features_nd[len(train_d):])

sample = random.sample(xrange(len(test_pred)),10)

for text,sentiment in zip(test_d.Text[sample],test_pred[sample]):
    print sentiment,text