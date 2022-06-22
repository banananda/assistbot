import nltk
import string
from nltk.stem import WordNetLemmatizer 
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import random 
import json
import streamlit as st


@st.cache
def nltk_init():
  nltk.download("punkt")
  nltk.download("wordnet")
  nltk.download('omw-1.4')

# nltk_init()

model = keras.models.load_model("./chatbot_model")
#model = keras.models.load_model(r"C:\Users\renan\OneDrive - Bina Nusantara\Documents\Semester 4\Natural Language Processing\stoopid-chatbot\chatbot_model")

f = open("./datasets/Intent.json")
#f = open(r"C:\Users\renan\OneDrive - Bina Nusantara\Documents\Semester 4\Natural Language Processing\stoopid-chatbot\datasets\Intent.json")
data = json.load(f)

lemmatizer = WordNetLemmatizer()

words = []
classes = []
doc_X = []
doc_y = []
# Loop through all the intents
# tokenize each pattern and append tokens to words, the patterns and
# the associated tag to their associated list
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])
    
    # add the tag to the classes if it's not there already 
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))


def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels): 
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

def get_response(intents_list, intents_json): 
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result

st.write('''
# Botbot
###### Made with love 💖
***
''')
message = st.text_input("Talk to me")

if(message == "stop"):
    isRun = False
else:
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    st.write("\n\nBotbot: " + result)
