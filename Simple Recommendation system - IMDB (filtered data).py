# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:55:40 2018

@author: santi

A recommendataion system implementation inspired by DataCamp tutorial written
by Rounak Banik

"""

# Recommendation System

###############################################################################
#                    Simple recommendation system
###############################################################################

'''
Build a simple recommendation system against the IMDB metadata set

1. Decide the metric or scoring approach to rate the movie
2. Calculate the score for all the movies
3. Sort the movies and recommend the movies according to the output result

The movies scores will be weighted according to the number of vote as well as 
the rates. In order to avoid the circumstances that popular movies being out 
voted by movies with only a few high rank votes, the movie score will be adjusted
as follow: 
    
    Weighted Rating = (vR/v+m) + (mC/v+m)
    
    - where v is the number of votes
    - m is the minimum votes required (divided by v+m to mutualize the effect of 
                                       less popular movies)
    - R is the average rating of the movie
    - C is the mean vote cross the whole report

'''
import pandas as pd

metadata = pd.read_csv('movies_metadata.csv', low_memory = False)

C = metadata.vote_average.mean()

# Select 90th percentale of number of votes as the threshold m
m = metadata.vote_count.quantile(0.9)

filtered_movies = metadata.copy().loc[metadata.vote_count >= m]

# Score the filtered movies

def get_score(X, m=m, C=C):
    
    # A function that calcuate the weighted score for the input movies
    v = X.vote_count
    R = X.vote_average    
    
    return (v*R/(v+m)) + (m*C/(v+m))

filtered_movies['score'] = filtered_movies.apply(get_score, axis = 1)

filtered_movies = filtered_movies.sort_values('score', ascending = False)

top_movies = filtered_movies[['title', 'vote_count', 'vote_average', 'score']]


###############################################################################
#          Description-Based(Item-Base) Recommendation System
###############################################################################

'''
Tokenize the comment of each movie with TF-IDF framework then calculate the 
similarity between them to identify the similar movies to for recommendation 
purpose

Several similarity metrics are available for this case such as euclidean distance, 
pearson and cosine similarity, etc. Cosine similarity is selected in this case 
to calcuate the similarity between two movies according to their comments. 

For computational efficiency purpose, linear_kernel is selected over 
cosine_similarities.  


Alteratively, word2vec scheme that captures more semantic meanings can be 
applied to improve the performance

'''

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the tfidf vectorizer and remove the common English stopwords
tfidf = TfidfVectorizer(stop_words = 'english')

# Replace the NA with empty string
#metadata.overview = metadata.overview.fillna('na')
filtered_movies.overview = filtered_movies.overview.fillna('')

# generate the tfidf matrix
#tfidf_meta = tfidf.fit_transform(metadata.overview)
tfidf_matrix = tfidf.fit_transform(filtered_movies.overview)

#tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

# Mind the size of the input matrix, converting the original metadata leads to
# an out of memeory issue

#cosine_sim = linear_kernel(tfidf_meta, tfidf_meta)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping series to map the movie titles to index in metadata
'''
NEED TO REINDEX THE FILTERED_MOVIES AND SET THE INDEX FROM (0,LEN(FILTERED_MOVIES))
'''
filtered_movies.index = pd.RangeIndex(len(filtered_movies))
indices = pd.Series(filtered_movies.index, index = filtered_movies.title).drop_duplicates()
#indices = pd.Series(filtered_movies.index, index = filtered_movies.title).drop_duplicates()

def content_based_recommender(title, indices = indices, cosine_sim = cosine_sim):
    
    # get the index of the given movie title
    idx = indices[title]
    
    # Get the pairwise similarity score of all the movies 
    sim_score = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_score = sorted(sim_score, key = lambda x: x[1], reverse = True)
    
    # Get the score of the top 10 similar movies
    similar_movies = sim_score[1:11]
    
    # Get the indices
    movie_indices = [i[0] for i in similar_movies]
    
    return filtered_movies['title'].iloc[movie_indices]


# Test the content based recommender system 
content_based_recommender("The Dark Knight Rises")



###############################################################################
#       Credits, Genres, and Keywords-Based Recommendation System
###############################################################################

# Load the credits and keywords
credit = pd.read_csv('credits.csv')
keyword = pd.read_csv('keywords.csv') 

# remove the observations with bad IDs
metadata = metadata.drop([19730, 29503, 35587])

filtered_movies_2 = metadata.copy().loc[metadata.vote_count >= m]


# Convert the id into int
filtered_movies_2.id = metadata.id.astype('int')
credit.id = credit.id.astype('int')
keyword.id = keyword.id.astype('int')

# Merge the credit and keyword with metadata
filtered_movies_2 = filtered_movies_2.merge(credit, on = 'id')
filtered_movies_2 = filtered_movies_2.merge(keyword, on = 'id')

#metadata = filtered_movies_2.copy()

from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:
    filtered_movies_2[feature] = filtered_movies_2[feature].apply(literal_eval)
    
import numpy as np

# Get the director's name
def get_director(x):
    
    for i in x:
        if i['job'] == 'Director':
            return i['name']
        
    return np.nan
    
# Get the list of top 3 actors
def get_list(x): 
    if isinstance(x, list): 
        names = [i['name'] for i in x]
        
        # Check if there are more than 3 actors/actresses
        if len(names) > 3:
            names = names[:3]
        
        return names
    
    # Return empty list for unmatched movies
    return []

# Define the newly extracted features 
filtered_movies_2['director'] = filtered_movies_2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']

for feature in features:
    filtered_movies_2[feature] = filtered_movies_2[feature].apply(get_list)
    
#metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    filtered_movies_2[feature] = filtered_movies_2[feature].apply(clean_data)
    
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

filtered_movies_2['soup'] = filtered_movies_2.apply(create_soup, axis=1)

# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filtered_movies_2['soup'])    
    
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of your main DataFrame and construct reverse mapping as before
#metadata = metadata.reset_index()
#indices = pd.Series(metadata.index, index=metadata['title'])

filtered_movies_2.index = pd.RangeIndex(len(filtered_movies_2))
indices_2 = pd.Series(filtered_movies_2.index, index = filtered_movies_2.title).drop_duplicates()

def keyword_based_recommender(title, indices = indices_2, cosine_sim = cosine_sim2):
    
    # get the index of the given movie title
    idx = indices[title]
    
    # Get the pairwise similarity score of all the movies 
    sim_score = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_score = sorted(sim_score, key = lambda x: x[1], reverse = True)
    
    # Get the score of the top 10 similar movies
    similar_movies = sim_score[1:11]
    
    # Get the indices
    movie_indices = [i[0] for i in similar_movies]
    
    return filtered_movies_2['title'].iloc[movie_indices]

keyword_based_recommender("The Dark Knight Rises")

keyword_based_recommender("The Godfather")
    
    
    
    
    
    
    
    

















