from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text1 = input("Enter first text: ")
text2 = input("Enter second text: ")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text1, text2])

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

percentage = similarity[0][0] * 100

print(f"Similarity Percentage: {percentage:.2f}%")
