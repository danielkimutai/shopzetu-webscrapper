import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("shopetu.csv")

# Title and description
st.title("Shopzetu simple recommender system")
st.write("""
##  Products
Below is the data scraped from shop zetu website""")
if st.button('List of Products'):
    st.write(df)

# Get user input
st.subheader("What would you like to purchase?")
product_name = st.text_input("Enter the product name:")

# Recommender function
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommender(product_name, data):
    # Check if the product exists in the dataset
    if product_name not in data["description"].values:
        print("Product not found. Please enter a valid product name.")
        return

    # Importing TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data["description"].fillna(""))

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get product index
    product_index = data[data["description"] == product_name].index[0]

    # Calculate similarity scores
    similarity_scores = pd.DataFrame(cosine_sim[product_index], index=data.index, columns=["score"])
    similarity_scores = similarity_scores.sort_values("score", ascending=False)

    # Get top 3 similar products (excluding the product itself)
    product_indices = similarity_scores.index[1:4]

    # Return recommended products
    recommended_products = data.loc[product_indices, ["description", "product_price", "ratings"]]
    return recommended_products


# print(recommended_products)

# Display recommended products
if product_name:
    recommended = recommender(product_name, df)
    if recommended is not None:
        st.write("You may also like:")
        st.table(recommended)