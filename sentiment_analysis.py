import pandas as pd
import re

# sklearn packages
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# nltk packages
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

RANDOM_SEED = 20


# data cleaning code
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if(text!=text):
        return ""
    text = text.lower()

    # removing all the symbols from tweets
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)

    # lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    text = [wordnet_lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)]
    text = ' '.join(word for word in text if word not in STOPWORDS)
    return text


# pipeline for text classification
from sklearn.svm import LinearSVC
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LinearSVC()),
              ])


# function to train the given model
def train_model():

    print('\nLoading data...')
    
    train = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding='latin',names=["target", "id", "date", "flag", "user", "text"])
    test = pd.read_csv('testdata.manual.2009.06.14.csv',encoding='latin',names=["target", "id", "date", "flag", "user", "text"])
    print('Training data loaded!')

    train.drop(['id','user','date','flag'], axis=1, inplace=True)
    test.drop(['id','user','date','flag'], axis=1, inplace=True)

    # applying clean_text() function
    print('\nPreprocessing the data...\nThis might take a few minutes...')
    train['text'] = train['text'].apply(clean_text)
    test['text'] = test['text'].apply(clean_text)
    print('Preprocessing complete!')

    X = train['text']
    y = train['target']

    X_test = test['text']
    y_test = test['target']

    valid_size = 0.3
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size,random_state=RANDOM_SEED)

    print("\nTraining Data Size:- ", int((X.shape[0])*(1-valid_size)))
    print("Validation Data Size:- ", int((X.shape[0])*(valid_size)))
    print("Test Data Size:- ", X_test.shape[0])

    print('\nTraining the model...')
    nb.fit(X_train[:200000], y_train[:200000])
    print('Training Complete!')

    y_pred = nb.predict(X_train)
    print('\nTraining Accuracy:-',100*accuracy_score(y_pred, y_train))
    print(classification_report(y_train, y_pred,zero_division=0))

    y_pred = nb.predict(X_valid)
    print('\nValidation Accuracy:-',100*accuracy_score(y_pred, y_valid))
    print(classification_report(y_valid, y_pred,zero_division=0))

    y_pred = nb.predict(X_test)
    print('\nTesting Accuracy:-',100*accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,zero_division=0))

def evaluate():
    tweet = [input('Please type a tweet: ')]
    sentiment = nb.predict(tweet)
    if sentiment[0]==4:
        print('\t>> It seems to be a Positive sentiment')
    elif sentiment[0]==2:
        print('\t>> It seems to be a Neutral sentiment')
    else:
        print('\t>>It seems to be a Negative sentiment')

train_model()

while True:
    option = int(input('Enter an option:\n1. To Evaluate a tweet\n2. Exit\n'))
    if option==1:
        evaluate()
    else:
        break