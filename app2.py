
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from sklearn.naive_bayes import MultinomialNB
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_table(url, header=None, names=['label', 'message'])

# Convert labels to binary values 
#(0 for ham, 1 for spam)
sms['label'] = np.where(sms['label']=='spam', 1, 0)

# Train the Naive Bayes classifier on the full dataset
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sms['message'])
y = sms['label']
clf = MultinomialNB()
clf.fit(X, y)

# Define a function to predict the label of a message
def predict(message):
  message_counts = vectorizer.transform([message])
  return clf.predict(message_counts)[0]

# Create a Streamlit app
col1, col2 = st.columns(2)
st.title('Spam Classifier')

with col1:
    # Create a canvas component
    if st.button('Enter a message'):
      message = st.text_input('Enter a message')

with col2:
    if st.button('Predict'):
      prediction = predict(message)
      if prediction == 1:
        st.error('This is a spam message')
      else:
        st.success('This is a legitimate message')
