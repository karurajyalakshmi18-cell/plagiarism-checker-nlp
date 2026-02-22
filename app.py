import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Plagiarism Checker (Mini NLP Project)")

text1 = st.text_area("Enter First Text")
text2 = st.text_area("Enter Second Text")

if st.button("Check Similarity"):
    if text1 and text2:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        percentage = similarity[0][0] * 100
        st.success(f"Similarity: {percentage:.2f}%")
    else:
        st.warning("Please enter both texts")
