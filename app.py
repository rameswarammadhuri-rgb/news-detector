import streamlit as st
import pickle

st.title("Fake News Detection App")

text = st.text_area("Enter news text")

if st.button("Check"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        st.success("Prediction complete")
