'''
Steps:
1. Data Loading
2. Data Preprocessing
3. Feature Extraction
4. Implementation of Cosine Similarity Algorithm
5. User inout
6. Generating output
'''

# importing dependencies
import numpy as np  # for maths
import pandas as pd  # for data handling
import difflib  # for finding closest match
from sklearn.feature_extraction.text import TfidfVectorizer  # for converting text to numerical feature vectors
from sklearn.metrics.pairwise import cosine_similarity  # for measuring the similarity between feature vectors

# loading data
try:
    movies_dataset = pd.read_csv('movies.csv', header=0)   #HEADER=0 -> Use the first row (index 0) as the header (so we can later use them as column names)
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

# processing the data
print(movies_dataset.head())   # getting an idea of the data 
print(movies_dataset.info()) 
print(movies_dataset.isnull().sum())  # checking if any column has any null value, if yes we will to adjust for it.

# selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','title','cast','director']
print(selected_features)

# replacing the null valuess with null string
for feature in selected_features:
  movies_dataset[feature] = movies_dataset[feature].fillna('')  #.fillna repluces null with empty string

# combining all the 6 selected features
combined_features = movies_dataset['genres']+' '+movies_dataset['keywords']+' '+movies_dataset['tagline']+' '+movies_dataset['title']+' '+movies_dataset['cast']+' '+movies_dataset['director']
print(combined_features)

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features) 
"""Explanation
-> Fit and transform the combined features into a sparse matrix of TF-IDF values
-> TF-IDF (Term Frequency-Inverse Document Frequency) is a technique that assigns weights to words based on their frequency in a document (TF) and how unique they are across a collection of documents (IDF). It highlights important terms in text by giving higher weights to rare but relevant words, making it useful for tasks like text representation, search engines, and document similarity.
-> Sparse representation saves memory and speeds up operations on the matrix.
"""
print(feature_vectors)  # (0, 243)      0.07630361845708403 -> means that in document 0, the term with index 243 in the vocabulary has a TF-IDF score of 0.0763.

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)  # computes the pairwise similarity between TF-IDF feature vectors, resulting in a similarity matrix where each entry indicates how similar two documents are based on their vector representation. 
# shape of this similarity matrix is (4803, 4803) with each entry bin the range -1 to 1

movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_dataset['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_dataset[movies_dataset.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_dataset[movies_dataset.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1