import pandas as pd
import numpy as np
import surprise
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Convert input ratings into a dataframe
def process_input(input_ratings, new_user_id):
    input_ratings = pd.DataFrame(input_ratings)
    
    input_ratings = input_ratings[(input_ratings['movieId'].notnull()) & (input_ratings['rating'].notnull()) & \
                              (input_ratings['rating'] != 'not seen')]
    input_ratings['movieId'] = input_ratings['movieId'].astype(int)
    
    input_ratings['userId'] = new_user_id
    
    return input_ratings

# Get the train and test set
def get_train_test_set(input_ratings_df, new_user_id,ratings, movies):
    
    # Add the input ratings to the ratings dataset
    trainset = pd.concat([ratings, input_ratings_df], axis = 0)
    trainset = trainset[['userId', 'movieId', 'rating']]
    
    # The testing set is all of the movies for the new user
    testset = movies.copy()
    
    testset['userId'] = new_user_id
    testset['rating'] = np.nan
    
    testset = testset[~testset['movieId'].isin(list(input_ratings_df['movieId']))]
    testset = testset[['userId', 'movieId', 'rating']]
    
    return trainset, testset

def fit_model(trainset, n_factors = 100, n_epochs = 20):
    reader = Reader(rating_scale=(0.5,5))
    
    trainset = Dataset.load_from_df(trainset, reader)
    trainset = trainset.build_full_trainset()
    
    algo = SVD(n_epochs, n_factors)
    algo.fit(trainset)
    return algo

def get_predictions(model, testset, ratings):
    predictions = model.test(testset = testset.values.tolist())
    predictions = pd.DataFrame([list(x) for x in predictions], 
             columns = ['userId', 'movieId', 'rating', 'predicted_rating', 'was_impossible'])

    num_ratings = pd.pivot_table(
        ratings, index = 'movieId', values = 'rating', aggfunc = 'count')

    num_ratings = num_ratings.rename(columns = {"rating": "num_ratings"})

    predicted_ratings = predictions.merge(num_ratings, left_on ='movieId', right_index = True)

    predicted_ratings = predicted_ratings.drop(columns = ['rating', 'was_impossible']).rename(
        columns = {'predicted_rating': 'rating'})
    
    return predicted_ratings

def get_movie_by_genre(predicted_ratings, genres, movies_genres, max_recommendations = 20):
    
    predictions_genres = predicted_ratings.merge(movies_genres, on = 'movieId')
    predictions_genres = predictions_genres.sort_values(by = ['genreId', 'adjusted_rating'], ascending=False)

    top100_predictions_genres = predictions_genres.groupby('genreId').head(max_recommendations)
    top100_predictions_genres = top100_predictions_genres.sort_values(by = 'adjusted_rating', ascending = False)
    return top100_predictions_genres