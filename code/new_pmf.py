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
eta = 0.01         #learning rate
lambdaU = 0.1
lambdaV = 0.1
maxRating = 5
als_iterations = 20
training_iterations = 100

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
    #validation_indices =  random.sample(range(0, len(training_list)), int(0.05*len(training_list)))
    #pickle.dump(validation_indices, open("validation_indices", "wb"))
    validation_indices = pickle.load(open("validation_indices", "rb"))
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
def get_dictionaries(userid_list, movieid_list, rating_list):
    number_users = max(userid_list)
    number_movies = max(movieid_list)
    # rating_list = normalize_ratings(rating_list)
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

def loss(U, V, userMovieDict):
    loss = 0
    product = U.T.dot(V)
    for i in range (0, product.shape[0]):
        for j in range(0, product.shape[1]):
            if (i+1) in userMovieDict and (j+1) in userMovieDict[i+1]:
                loss = loss + (userMovieDict[i+1][j+1] - product[i][j]) ** 2

    loss = loss * 1.0/2
    loss = loss +  (lambdaU * 1.0/2) * (LA.norm(U, 'fro') ** 2)
    loss = loss + (lambdaV * 1.0/2) * (LA.norm(V, 'fro') ** 2)
    return loss


def RMSE(predicts, actual):
    rmse = 0.0
    counter = 0
    for user in actual:
    	for movie in actual[user]:
    	    rmse = rmse + (actual[user][movie]-predicts[user - 1][movie - 1])**2
            counter = counter + 1
    rmse = (rmse * 1.0/counter) ** 0.5
    return rmse


def ALS(U, V, ratings_matrix):

    for i in range(als_iterations):
        # update U, latent vector U, fixed vector V
        VTV = V.dot(V.T)    # D * D
        lambdaU_matrix = np.eye(VTV.shape[0]) * lambdaU
        for u in xrange(U.shape[1]):
            U[:, u] = solve((VTV + lambdaU_matrix), ratings_matrix[u, :].T.dot(V.T))

        # update V
        UTU = U.dot(U.T)
        lambdaV_matrix = np.eye(UTU.shape[0]) * lambdaV
        for v in xrange(V.shape[1]):
            V[:, v] = solve((UTU + lambdaV_matrix), ratings_matrix[:, v].T.dot(U.T))
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
    ratings_matrix = np.random.uniform(low=1.0, high=5.0, size=(number_users + 1, number_movies + 1))
    valid_ratings_matrix = np.random.uniform(low=1.0, high=5.0, size=(valid_number_users + 1, valid_number_movies + 1))

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

    # U (D, 6040), V(D, 3883)
    U = np.random.rand(D, number_users)
    V = np.random.rand(D, number_movies)

    # U = pickle.load(open("U", "rb"))
    # V = pickle.load(open("V", "rb"))

    for i in range(training_iterations):
        U, V = ALS(U, V, ratings_matrix)
        predictions = U.T.dot(V)
        # TODO:
        training_loss = loss(U, V, userMovieDict)
        training_RMSE = RMSE(predictions, userMovieDict)

        valid_loss = loss(U, V, valid_user_movie_dict)
        valid_RMSE = RMSE(predictions, valid_user_movie_dict)
        print '##### training iterations %d ####' % (i)
        print "Train Loss ", training_loss
        print "Train RMSE ", training_RMSE
        print "Valid Loss ", valid_loss
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
