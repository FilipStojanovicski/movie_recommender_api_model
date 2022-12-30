import pandas as pd

# Convert input ratings into a matrix
def process_input(input_ratings):
    input_ratings = pd.DataFrame(input_ratings)
    
    input_ratings = input_ratings[(input_ratings['movieId'].notnull()) & (input_ratings['rating'].notnull()) & \
                              (input_ratings['rating'] != 'not seen')]
    input_ratings['movieId'] = input_ratings['movieId'].astype(int)
    
    input_ratings_matrix = pd.pivot_table(
    data = input_ratings, columns = 'movieId', values='rating', aggfunc = 'first')
    
    return input_ratings_matrix.iloc[0]

# Find most similar users
def get_nearest_neighbors(input_ratings_matrix, ratings_by_user, min_support, max_neighbours):
    ratings_diffs = (ratings_by_user - input_ratings_matrix)**2
    user_dists = ratings_diffs.sum(axis='columns', min_count = min_support)
    user_dists = user_dists.dropna()

    print("Num of users with min support movies rated")
    print(len(user_dists))
    if len(user_dists) == 0:
        return None
    user_sims = 1 / (1 + user_dists)
    user_sims = pd.DataFrame(user_sims, columns = ['similarity']).sort_values(
    by = 'similarity', ascending = False)
    most_similar_users = user_sims.iloc[0:max_neighbours]
    return most_similar_users


def ratings_predictions_no_neighbors(ratings_by_user, movies_rated):
    
    similar_user_ratings = ratings_by_user

    movies_to_rate = [x for x in list(similar_user_ratings.columns)  if (x not in movies_rated)]
        
    # Get the number of ratings for each movie
    num_ratings = pd.DataFrame(similar_user_ratings[movies_to_rate].notnull().sum(), columns = ['num_ratings'])
    # Calculate the mean weighted sum rating
    movie_ratings = pd.DataFrame(similar_user_ratings[movies_to_rate].mean(), columns = ['rating'])
    movie_ratings = pd.merge(movie_ratings, num_ratings, left_index = True, right_index = True)
    
    return movie_ratings

def ratings_predictions(most_similar_users, ratings_by_user, movies_rated):
    
    similar_user_ratings = pd.merge(ratings_by_user, most_similar_users, 
                                left_index = True, right_index = True)
    # Remove all movies that no one has rated
    similar_user_ratings = similar_user_ratings.dropna(axis = 1, how = 'all')

    print("Similar User Ratings")
    print(len(similar_user_ratings))
    

    movies_to_rate = [x for x in list(similar_user_ratings.columns)  if (x != 'similarity' and x not in movies_rated)]
    # For each movie get a weighted rating for each user
    for col in movies_to_rate:
        # Weight all of the ratings by the user similarity
        similar_user_ratings[col] = similar_user_ratings[col] * similar_user_ratings['similarity']
        total_weights = similar_user_ratings[similar_user_ratings[col].notnull()]['similarity'].sum()
        similar_user_ratings[col] = similar_user_ratings[col]/total_weights
        
    # Get the number of ratings for each movie
    num_ratings = pd.DataFrame(similar_user_ratings[movies_to_rate].notnull().sum(), columns = ['num_ratings'])
    # Calculate the mean weighted sum rating
    movie_ratings = pd.DataFrame(similar_user_ratings[movies_to_rate].sum(), columns = ['rating'])
    movie_ratings = pd.merge(movie_ratings, num_ratings, left_index = True, right_index = True)
    
    return movie_ratings

def get_movie_by_genre(predicted_ratings, genres, movies_genres, max_recommendations = 20):
    
    predictions_genres = predicted_ratings.merge(movies_genres, left_index = True, right_on = 'movieId')
    predictions_genres = predictions_genres.sort_values(by = ['genreId', 'adjusted_rating'], ascending=False)

    top100_predictions_genres = predictions_genres.groupby('genreId').head(max_recommendations)
    top100_predictions_genres = top100_predictions_genres.sort_values(by = 'adjusted_rating', ascending = False)

    return top100_predictions_genres