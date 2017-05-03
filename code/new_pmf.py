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
D = 7             #number of factors [1:20]
lambdaU = 0.4
lambdaV = 0.4
maxRating = 5
training_iterations = 20

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
# causes split such that 95% data is training
# and 5% data is for validation
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
    validation_indices =  random.sample(range(0, len(training_list)), int(0.05*len(training_list)))
    pickle.dump(validation_indices, open("validation_indices", "wb"))
    #validation_indices = pickle.load(open("validation_indices", "rb"))
    for index in validation_indices:
        validation_data.append(training_list[index])
    training_data = np.reshape(training_list, (1, len(training_list)))
    training_data = np.delete(training_data, validation_indices)
    training_data = list(training_data)
    return training_data, validation_data

'''
Build dictionaries for user-movie ratings
key: user_id
value: (dict)
    key: movie_id the user has rated
    value: ratings
'''
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

# Normalises ratings to [0,1] scale
def normalize_ratings(ratings):
    '''
    Input: ratings: ratings list
                 K: The upper bound of ratings
    Output: normalized value of ratings.[0, 1]
    '''
    K = maxRating
    for i in range(0, len(ratings)):
        ratings[i] = float((ratings[i] - 1)) / (K - 1)
    return ratings

def nonnormalize_ratings(ratings):
    for i in range(0, ratings.shape[0]):
        for j in range (0, ratings.shape[1]):
            if (ratings[i][j] > 5):
                    ratings[i][j] = 5
            if (ratings[i][j] < 1):
                ratings[i][j] = 1
    return ratings

def loss(U, V, ratings_matrix, w_matrix):
    product_matrix = U.T.dot(V)
    difference_matrix = ratings_matrix - product_matrix
    actual_difference_matrix = np.multiply(w_matrix, difference_matrix)
    square_matrix = np.square(actual_difference_matrix)
    print square_matrix.shape

    loss = np.sum(square_matrix)
    loss = loss +  (lambdaU  * (LA.norm(U, 'fro')))
    loss = loss + (lambdaV * (LA.norm(V, 'fro')))


def RMSE(predicts, actual):
    rmse = 0.0
    counter = 0
    for user in actual:
    	for movie in actual[user]:
    	    rmse = rmse + (actual[user][movie]-predicts[user - 1][movie - 1])**2
            counter = counter + 1
    rmse = (rmse * 1.0/counter) ** 0.5
    return rmse


def ALS(U, V, ratings_matrix, users_per_movie, movies_per_user):
    # update U, latent vector U, fixed vector V
    #VTV = (V.dot(w_movies)).dot(V.T)    # D * D
    #lambdaU_matrix = np.eye(VTV.shape[0]) * lambdaU
    lambdaU_matrix = np.eye(D) * lambdaU
    for u in range(0, U.shape[1]):
        if  u+1 not in movies_per_user:
            #print u+1
            continue
        V_sub = V[:, movies_per_user[u+1]]
        VTV = V_sub.dot(V_sub.T)
        ratings_matrix_u = ratings_matrix[u, :][movies_per_user[u+1]]
        second = V_sub.dot(ratings_matrix_u)
        U[:, u] = solve((VTV + lambdaU_matrix), second)
    
    # update V
    #UTU = (U.dot(w_users)).dot(U.T)
    #lambdaV_matrix = np.eye(UTU.shape[0]) * lambdaV
    lambdaV_matrix = np.eye(D) * lambdaV
    for v in range(0, V.shape[1]):
        if v+1 not in users_per_movie:
            #print "Movie No one rated: ", v+1
            continue
        U_sub = U[:, users_per_movie[v+1]]
        UTU = U_sub.dot(U_sub.T)
        ratings_matrix_v = ratings_matrix[:,v][users_per_movie[v+1]]
        second = U_sub.dot(ratings_matrix_v)
        V[:, v] = solve((UTU + lambdaV_matrix), second)
    pickle.dump(U, open("U", "wb"))
    pickle.dump(V, open("V", "wb"))

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
    w_matrix = np.zeros((number_users + 1, number_movies + 1))
    valid_ratings_matrix = np.zeros((valid_number_users + 1, valid_number_movies + 1))
    w_valid_matrix = np.zeros((valid_number_users + 1, valid_number_movies + 1))

    # training_rating_matrix
    for userid in userMovieDict:
        for movieid in userMovieDict[userid]:
            ratings_matrix[userid][movieid] = userMovieDict[userid][movieid]
            w_matrix[userid][movieid] = 1

    # valid_rating_matrix
    for userid in valid_user_movie_dict:
        for movieid in valid_user_movie_dict[userid]:
            valid_ratings_matrix[userid][movieid] = valid_user_movie_dict[userid][movieid]
            w_valid_matrix[userid][movieid] = 1

    # complete_ratings_matrix: size (6040, 3883)
    ratings_matrix = np.delete(ratings_matrix, 0, 0)
    ratings_matrix = np.delete(ratings_matrix, 0, 1)
    w_matrix = np.delete(w_matrix, 0, 0)
    w_matrix = np.delete(w_matrix, 0, 1)

    # valid_ratings_matrix
    valid_ratings_matrix = np.delete(valid_ratings_matrix, 0, 0)
    valid_ratings_matrix = np.delete(valid_ratings_matrix, 0, 1)
    w_valid_matrix = np.delete(w_valid_matrix, 0, 0)
    w_valid_matrix = np.delete(w_valid_matrix, 0, 1)

    users_per_movie, movies_per_user = get_users_per_movie_and_vice_versa(training_userid_list, training_movieid_list)

    # U (D, 6040), V(D, 3883)
    U = np.random.rand(D, number_users)
    V = np.random.rand(D, number_movies)

    # U = pickle.load(open("U", "rb"))
    # V = pickle.load(open("V", "rb"))

    for i in range(0, training_iterations):
        U, V = ALS(U, V, ratings_matrix, users_per_movie, movies_per_user )
        predictions = U.T.dot(V)
        #training_loss = loss(U, V, ratings_matrix, w_matrix)
        training_RMSE = RMSE(predictions, userMovieDict)

        #valid_loss = loss(U, V, valid_ratings_matrix, w_valid_matrix)
        valid_RMSE = RMSE(predictions, valid_user_movie_dict)
        print '##### training iterations %d ####' % (i)
        #print "Train Loss ", training_loss
        print "Train RMSE ", training_RMSE
        #print "Valid Loss ", valid_loss
        print "Valid RMSE ", valid_RMSE
        print ""

    #TESTING CODE FOLLOWS
    userid_list, movieid_list = preprocess_test_file(test_file)
    output = open(output_file, 'w')
    U = pickle.load(open("U", "rb"))
    V = pickle.load(open("V", "rb"))
    ratings = U.T.dot(V)
    ratings = nonnormalize_ratings(ratings)
    for i in range(0, len(userid_list)):
       user = userid_list[i]
       movie = movieid_list[i]
       output.write(str(ratings[user - 1][movie - 1]))
       output.write("\n")
    output.close()
    #    print ratings[user - 1][movie - 1]

main()
