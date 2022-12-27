from flask import Flask,jsonify,request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np

app = Flask(__name__)

db_username = 'filip'
db_password = 'abc123'
db_name = 'movie_recommender'
db_url = 'localhost'
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql://{db_username}:{db_password}@{db_url}/{db_name}'
db = SQLAlchemy(app)

def read_all_data():
    ratings = pd.read_sql(sql = "SELECT * FROM RATINGS", con=db.engine)
    ratings_by_user = pd.pivot_table(data = ratings, index = "userId", values="rating", columns="movieId")
    movies = pd.read_sql(sql = "SELECT * FROM MOVIES", con=db.engine)
    genres = pd.read_sql(sql = "SELECT * FROM genres", con=db.engine)
    movies_genres = pd.read_sql(sql = "SELECT * FROM movies_genres", con=db.engine)
    return((ratings,ratings_by_user, movies, genres, movies_genres))

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

@app.route('/movie_recommendation', methods = ['GET', 'POST'])
def recommendMovies():
    print(request.get_json())
    if(request.method=='POST'):

        (ratings,ratings_by_user, movies, genres, movies_genres) = read_all_data()

        req = request.get_json()

        movies_rated = [x['movieId'] for x in req['ratings']]

        min_support = req['min_support']
        max_neighbours = req['num_neighbors']

        input_ratings_matrix = process_input(req['ratings'])

        # How many matching movies need to have been rated to be considered as a neighbor
        num_movies_rated = len(input_ratings_matrix)
        min_support = min(num_movies_rated, min_support)

        print(min_support)

        try:

            most_similar_users = get_nearest_neighbors(input_ratings_matrix, ratings_by_user, min_support, max_neighbours)

            if (most_similar_users is not None and (len(most_similar_users) > 0)):
                print("has neighbors")
                predicted_ratings = ratings_predictions(most_similar_users, ratings_by_user, movies_rated)
            else: 
                print("no neighbors")
                predicted_ratings = ratings_predictions_no_neighbors(ratings_by_user, movies_rated)    

            # Get the relative confidence score, the more total ratings, the more confident we can be in the score
            predicted_ratings['relative_confidence'] = np.log(1 + predicted_ratings['num_ratings'])
            # Weight the rating with the confidence
            predicted_ratings['adjusted_rating'] = predicted_ratings['rating'] * predicted_ratings['relative_confidence']

            predicted_ratings = predicted_ratings.sort_values(by = 'adjusted_rating', ascending = False)
            
            predictions_genres = get_movie_by_genre(predicted_ratings, genres, movies_genres)
            
            response = pd.merge(predictions_genres, movies[['movieId','title', 'genres']], left_on = 'movieId', right_on = 'movieId')
        
            response = response[['rating', 'adjusted_rating', 'movieId', 'genreId', 'title']].to_dict(
                orient='records')
                  
            return(jsonify({"status": 200, "data": response}))

        except Exception as e:
            print(e)
            return(jsonify({"status": 400, "data": None}))
      

    elif(request.method == 'GET'):
        print("GET REQUEST")
        data = {
            "ERROR" : "Invalid request"
        }
        return jsonify(data)
        
  
if __name__=='__main__':
    app.run(debug=True)