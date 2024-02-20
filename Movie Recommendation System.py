# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 02:39:53 2024

@author: Dell
"""
#################Importing libraries
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error

##################Load dataset
df =  pd.read_csv('movies.csv')

################# Description of data
df.head()
df.shape
df.info()
df.columns
df.describe()

############### Preprocessing of data

# checking for missing values
df.isnull().sum()


# null values in heat map
sns.heatmap(df.isnull(),cbar = False)

# check for duplicate values
df.duplicated().sum()


sns.displot(data=df, x='Ratings', kind='kde', aspect=1, height=4)

# Convert the dataframe into a pivot table
user_item_matrix = pd.pivot_table(df,index=df.index, columns='movieId', values='Ratings', fill_value=0)


user_item_matrix

# Convert the pivot table to a sparse matrix
user_item_matrix = csr_matrix(user_item_matrix.values)

user_item_matrix



# model training
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_item_matrix)



def get_movie_recommendations(movie_id, num_recommendations):
    # Find the k nearest neighbors of the movie
    distances, indices = model.kneighbors(user_item_matrix[movie_id], n_neighbors=num_recommendations+1)

    # indices of the recommended movies
    recommended_movie_indices = indices.squeeze()[1:]

    # titles of the recommended movies
    recommended_movie_titles = df.loc[recommended_movie_indices, 'title']

    return recommended_movie_titles

# Test the model by getting recommendations for a movie
movie_id = int(input("Enter Movie ID: "))
num_recom=int(input("how many movie recommendations you want? = "))
print("-----------------------------------------------------")
print("here is the list of",num_recom,"movies for you based on movie_id",movie_id)
print("-----------------------------------------------------")
recommended_movies = get_movie_recommendations(movie_id,num_recom)
print(recommended_movies)




# Assume the following are the actual ratings for the recommended movies by a user
actual_ratings = [0, 3, 2, 0, 1]

# Calculate the mean absolute error and root mean squared error
predicted_ratings = [df.loc[df['title'] == title, 'Ratings'].mean() for title in recommended_movies]

mae = mean_absolute_error(actual_ratings, predicted_ratings)
rmse = mean_squared_error(actual_ratings, predicted_ratings, squared=False)

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')


# accuracy  of the model
accuracy = 1 - (mae / max(actual_ratings))

print(f'Accuracy: {accuracy * 100}%')