from flask import Flask,jsonify,request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import model_knn, model_svd
import math
from flask import json
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

db_username = 'filip'
db_password = 'abc123'
db_name = 'movie_recommender'
db_url = '127.0.0.1'
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql://{db_username}:{db_password}@{db_url}/{db_name}'
db = SQLAlchemy(app)


def read_all_data():
    ratings = pd.read_sql(sql = "SELECT * FROM ratings", con=db.engine)
    ratings_by_user = pd.pivot_table(data = ratings, index = "userId", values="rating", columns="movieId")
    movies = pd.read_sql(sql = "SELECT * FROM movies", con=db.engine)
    genres = pd.read_sql(sql = "SELECT * FROM genres", con=db.engine)
    movies_genres = pd.read_sql(sql = "SELECT * FROM movies_genres", con=db.engine)
    return((ratings,ratings_by_user, movies, genres, movies_genres))

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    print(e)
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "status": 500,
        "data": {
            "code": e.code,
            "name": e.name,
            "description": e.description,
        }
    })
    print(response.status)
    response.content_type = "application/json"
    return response

@app.route('/movie_recommendation/knn', methods = ['GET', 'POST'])
def recommendMoviesKNN():
    print(request.get_json())
    if(request.method=='POST'):
        # try:

            (ratings,ratings_by_user, movies, genres, movies_genres) = read_all_data()

            req = request.get_json()

            movies_rated = [x['movieId'] for x in req['ratings']]

            min_support = req['model_params']['min_support']
            max_neighbours = req['model_params']['num_neighbors']
            popularity_boost = req['model_params']['popularity_boost']

            input_ratings_matrix = model_knn.process_input(req['ratings'])

            # How many matching movies need to have been rated to be considered as a neighbor
            num_movies_rated = len(input_ratings_matrix)
            min_support = min(num_movies_rated, min_support)

            print(min_support)

            most_similar_users = model_knn.get_nearest_neighbors(input_ratings_matrix, ratings_by_user, min_support, max_neighbours)

            if (most_similar_users is not None and (len(most_similar_users) > 0)):
                print("has neighbors")
                predicted_ratings = model_knn.ratings_predictions(most_similar_users, ratings_by_user, movies_rated)
            else: 
                print("no neighbors")
                predicted_ratings = model_knn.ratings_predictions_no_neighbors(ratings_by_user, movies_rated)    

            # Get the relative confidence score, the more total ratings, the more confident we can be in the score
            if (popularity_boost > 1):
                predicted_ratings['relative_confidence'] = (1 + predicted_ratings['num_ratings']).apply(lambda x: math.log(x, popularity_boost))
            else:
                predicted_ratings['relative_confidence'] = 1

            # Weight the rating with the confidence
            predicted_ratings['adjusted_rating'] = predicted_ratings['rating'] * predicted_ratings['relative_confidence']

            predicted_ratings = predicted_ratings.sort_values(by = 'adjusted_rating', ascending = False)
            
            predictions_genres = model_knn.get_movie_by_genre(predicted_ratings, genres, movies_genres)
            
            response = pd.merge(predictions_genres, movies[['movieId','title', 'genres']], left_on = 'movieId', right_on = 'movieId')
        
            response = response[['rating', 'adjusted_rating', 'movieId', 'genreId', 'title']].to_dict(
                orient='records')
                
            return(jsonify({"status": 200, "data": response}))

        # except Exception as e:
        #     print(e)
        #     return(jsonify({"status": 400, "data": None}))
    elif(request.method == 'GET'):
        print("GET REQUEST")
        data = {
            "Error" : "GET Request not supported for /movie_recommendation"
        }
        return jsonify(data)
        
  
if __name__=='__main__':
    app.run(debug=True)

@app.route('/movie_recommendation/svd', methods = ['GET', 'POST'])
def recommendMoviesSVD():
    print(request.get_json())
    if(request.method=='POST'):

        # try:

            (ratings, ratings_by_user, movies, genres, movies_genres) = read_all_data()

            req = request.get_json()

            movies_rated = [x['movieId'] for x in req['ratings']]

            n_factors = req['model_params']['n_factors']
            n_epochs = req['model_params']['n_epochs']
            popularity_boost = req['model_params']['popularity_boost']

            print(req['ratings'])

            if (len(movies_rated) > 0):

                # Create new user id
                new_user_id = ratings['userId'].max() + 1

                # Process input ratings
                input_ratings_df = model_svd.process_input(req['ratings'], new_user_id)

                # Get train and test set
                trainset, testset = model_svd.get_train_test_set(input_ratings_df, new_user_id, ratings, movies)

                # Create model
                model = model_svd.fit_model(trainset, n_factors, n_epochs)

                predicted_ratings = model_svd.get_predictions(model, testset, ratings)

            else: 
                predicted_ratings = model_knn.ratings_predictions_no_neighbors(ratings_by_user, movies_rated)   

            # Get the relative confidence score, the more total ratings, the more confident we can be in the score
            if (popularity_boost > 1):
                predicted_ratings['relative_confidence'] = (1 + predicted_ratings['num_ratings']).apply(lambda x: math.log(x, popularity_boost))
            else:
                predicted_ratings['relative_confidence'] = 1

            # Weight the rating with the confidence
            predicted_ratings['adjusted_rating'] = predicted_ratings['rating'] * predicted_ratings['relative_confidence']

            predicted_ratings = predicted_ratings.sort_values(by = 'adjusted_rating', ascending = False)

            predictions_genres = model_svd.get_movie_by_genre(predicted_ratings, genres, movies_genres)

            response = pd.merge(predictions_genres, movies[['movieId','title', 'genres']], left_on = 'movieId', right_on = 'movieId')
            
            response = response[['rating', 'adjusted_rating', 'movieId', 'genreId', 'title']].to_dict(
                orient='records')
                    
            return(jsonify({"status": 200, "data": response}))
        
        # except Exception as e:
        #     print(e)
        #     return(jsonify({"status": 400, "data": None}))

    elif(request.method == 'GET'):
        print("GET REQUEST")
        data = {
            "Error" : "Invalid request"
        }
        return jsonify(data)
        
  
if __name__=='__main__':
    app.run(debug=True)