import os
from scipy.sparse import coo_matrix
from numpy import linalg as LA
import numpy as np
import time
from scipy.special import expit
import random
import pickle
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error

data_dir = os.path.join('../', 'RSdata/')
training_file = os.path.join(data_dir, "training_rating.dat")
test_file = os.path.join(data_dir, "testing.dat")
output_file = os.path.join('./', "results.csv")

'''
Tunable parameters
'''
D = 5             
lambdaVal = 0.45
maxRating = 5
training_iterations = 30
valid_percentage = 0.05

def preprocess_test_file(test_file):
    movieid_list = []
    userid_list = []
    with open(test_file, "r") as test_data:
        for line in test_data:
            elements = line.rstrip("\n").split(" ")
            userid_list.append(int(elements[0]))
            movieid_list.append(int(elements[1]))
        test_data.close()
    return userid_list, movieid_list

# Splits each line into userid, movieid and rating
# Generate lists for these three items
def preprocess_training_file(training_data):
    userid_list = []
    movieid_list = []
    rating_list = []

    for line in training_data:
        elements = line.rstrip("\n").split("::")
        userid_list.append(int(elements[0]))
        movieid_list.append(int(elements[1]))
        rating_list.append(int(elements[2]))

    return userid_list, movieid_list, rating_list

# Removes bad lines from training file and
# causes split into training and testing
def split_training_data(original_training_file):
    training_list = []
    validation_data = []
    with open(original_training_file, "r") as training_data:
        for line in training_data:
            elements = line.rstrip("\n").split("::")
            if has_empty(elements):
                continue
            training_list.append(line)
    training_data.close()
    validation_indices =  random.sample(range(0, len(training_list)), int(valid_percentage*len(training_list)))
    for index in validation_indices:
        validation_data.append(training_list[index])
    training_data = np.reshape(training_list, (1, len(training_list)))
    training_data = np.delete(training_data, validation_indices)
    training_data = list(training_data)
    return training_data, validation_data

def get_users_per_movie_and_vice_versa(userid_list, movieid_list):
    users_per_movie = dict()
    movies_per_user = dict()
    for i in range(len(userid_list)):
        user = userid_list[i]
        movie = movieid_list[i]
        list_users = []
        list_movies = []
        if movie in users_per_movie:
            list_users = users_per_movie[movie]
        if user in movies_per_user:
            list_movies = movies_per_user[user]
        list_users.append(user)
        list_movies.append(movie)
        users_per_movie[movie] = list_users
        movies_per_user[user] = list_movies

    for movie in users_per_movie:
        list_users = users_per_movie[movie]
        list_users.sort()
        list_users[:] = [x - 1 for x in list_users]
        users_per_movie[movie] = list_users

    for user in movies_per_user:
        list_movies = movies_per_user[user]
        list_movies.sort()
        list_movies[:] = [x - 1 for x in list_movies]
        movies_per_user[user] = list_movies

    return users_per_movie, movies_per_user

'''
Build dictionaries for user-movie ratings
key: user_id
value: (dict)
    key: movie_id the user has rated
    value: ratings
'''
def get_dictionaries(userid_list, movieid_list, rating_list):
    number_users = max(userid_list)
    number_movies = max(movieid_list)
    userMovieDict  = dict()
    for i in range(len(userid_list)):
        user = userid_list[i]
        movie = movieid_list[i]
        movieRatingsDict = dict()
        if user in userMovieDict:
            movieRatingsDict = userMovieDict[user]
        movieRatingsDict[movie] = rating_list[i]
        userMovieDict[user] = movieRatingsDict
    return userMovieDict, number_users, number_movies

# Helper function to remove bad lines
def has_empty(elements):
    '''
    Output: if there is empty elements, return True else False
    '''
    for e in elements:
        if not e.strip():
            return True
    return False

def RMSE(predicts, actual):
    rmse = 0.0
    counter = 0
    for user in actual:
    	for movie in actual[user]:
            prediction = predicts[user - 1][movie -1]
            if (prediction < 1.0):
                prediction = 1.0
            elif (prediction > 5.0):
                prediction = 5.0       
    	    rmse = rmse + (actual[user][movie]-prediction)**2
            counter = counter + 1
    rmse = (rmse * 1.0/counter) ** 0.5
    return rmse


def ALS(U, V, ratings_matrix, users_per_movie, movies_per_user):
    # update U, latent vector U, fixed vector V
    lambdaU_matrix = np.eye(D) * lambdaVal
    for u in range(0, U.shape[1]):
        if  u+1 not in movies_per_user:
            continue
        V_sub = V[:, movies_per_user[u+1]]
        VTV = V_sub.dot(V_sub.T)
        ratings_matrix_u = ratings_matrix[u, :][movies_per_user[u+1]]
        second = V_sub.dot(ratings_matrix_u)
        U[:, u] = solve((VTV + lambdaU_matrix), second)
    
    # update V
    lambdaV_matrix = np.eye(D) * lambdaVal
    for v in range(0, V.shape[1]):
        if v+1 not in users_per_movie:
            continue
        U_sub = U[:, users_per_movie[v+1]]
        UTU = U_sub.dot(U_sub.T)
        ratings_matrix_v = ratings_matrix[:,v][users_per_movie[v+1]]
        second = U_sub.dot(ratings_matrix_v)
        V[:, v] = solve((UTU + lambdaV_matrix), second)
    #pickle.dump(U, open("U", "wb"))
    #pickle.dump(V, open("V", "wb"))

    return U, V

'''
for the names that don't have a "training" or have training
they all refer to training data
'''
def main():
    training_data, validation_data = split_training_data(training_file)
    training_userid_list, training_movieid_list, training_rating_list = preprocess_training_file(training_data)
    valid_userid_list, valid_movieid_list, valid_rating_list = preprocess_training_file(validation_data)

    userMovieDict, number_users, number_movies = get_dictionaries(training_userid_list, training_movieid_list, training_rating_list)
    valid_user_movie_dict, valid_number_users, valid_number_movies = get_dictionaries(valid_userid_list, valid_movieid_list, valid_rating_list)

    # (number_users, number_movies) (6040, 3883)
    ratings_matrix = np.zeros((number_users + 1, number_movies + 1))
    valid_ratings_matrix = np.zeros((valid_number_users + 1, valid_number_movies + 1))

    # training_rating_matrix
    for userid in userMovieDict:
        for movieid in userMovieDict[userid]:
            ratings_matrix[userid][movieid] = userMovieDict[userid][movieid]

    # valid_rating_matrix
    for userid in valid_user_movie_dict:
        for movieid in valid_user_movie_dict[userid]:
            valid_ratings_matrix[userid][movieid] = valid_user_movie_dict[userid][movieid]

    # complete_ratings_matrix: size (6040, 3883)
    ratings_matrix = np.delete(ratings_matrix, 0, 0)
    ratings_matrix = np.delete(ratings_matrix, 0, 1)

    # valid_ratings_matrix
    valid_ratings_matrix = np.delete(valid_ratings_matrix, 0, 0)
    valid_ratings_matrix = np.delete(valid_ratings_matrix, 0, 1)

    users_per_movie, movies_per_user = get_users_per_movie_and_vice_versa(training_userid_list, training_movieid_list)
    
    averageRating = sum(training_rating_list) * 1.0/len(training_rating_list)
 
    U = np.ones((D, number_users)) * averageRating
    V = np.ones((D, number_movies)) * averageRating

    for i in range(0, training_iterations):
        U, V = ALS(U, V, ratings_matrix, users_per_movie, movies_per_user )
        predictions = U.T.dot(V)
        training_RMSE = RMSE(predictions, userMovieDict)
        valid_RMSE = RMSE(predictions, valid_user_movie_dict)
        print '##### training iterations %d ####' % (i)
        print "Train RMSE ", training_RMSE
        print "Valid RMSE ", valid_RMSE
        print ""
    
    #TESTING CODE FOLLOWS
    userid_list, movieid_list = preprocess_test_file(test_file)
    output = open(output_file, 'w')
    ratings = U.T.dot(V)
    for i in range(0, len(userid_list)):
       user = userid_list[i]
       movie = movieid_list[i]
       rating = ratings[user - 1][movie - 1]
       if (rating < 1):
           rating = 1.0
       elif (rating > 5):
           rating = 5.0
       output.write(str(rating))
       output.write("\n")
    output.close()
main()
