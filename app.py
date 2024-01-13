import streamlit as st
import time
import tensorflow as tf
import keras
import numpy as np



model = keras.models.load_model('./model/bert_model.keras' )



def predict(test_string):
  class_prob = model.predict(test_string, batch_size=1)[0]
  
  return class_prob


st.title("Sentiments from Text")

text = st.text_input("Enter text here..")
text = text.apply(str.lower)

if st.button('Check Sentiments'):
  with st.status("Checking vibe...") as status:
    st.write("Searching for Sentiments...")
    time.sleep(2)
   
    prediction = predict(text)
    if np.argmax(prediction) == 0:
       st.warning("This shows Negative Sentiments")
    elif np.argmax(prediction) == 1:
       st.info("This shows Neutral Sentiments")
    else:
       st.success("This shows Positive Sentiments")
    status.update(label="Download complete!", state="complete", expanded=False)
  