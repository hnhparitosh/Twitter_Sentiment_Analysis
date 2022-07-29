# Twitter_Sentiment_Analysis

This is a simple ML model that is trained to predict the sentiment depicted by a tweet.  
The dataset used for training was taken from this [link](https://drive.google.com/file/d/1t6iUlMZamG5kpf_ms6o-lgHzSOOgpXTP).  

The model was trained in Python using four modules: sklearn, nltk, re and pandas.  
Python file sentiment_analysis.py contains all the clean code to be used, although jupyter notebook (unclean code) is also present.  

Model was trained using sklearn pipeline: CountVectorizer, TfidfTransformer and LinearSVC classifier.  
The training data contained only 2 labels - Positive and Negative Sentiment but test data also had Neutral Sentiment.  
Thats why accuracy score was better for train and validation set but less for test set.  

Run the sentiment_analysis.py to train the model and Evaluate sentiment for any tweet.  

If you like this project, please give this repo a ⭐.
