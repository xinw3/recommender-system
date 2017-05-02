import os
from scipy.sparse import coo_matrix
from sklearn import preprocessing
from numpy import linalg as LA
import numpy as np
import time
from scipy.special import expit
import random
import pickle
from numpy.linalg import solve

data_dir = os.path.join('../', 'RSdata/')
training_file = os.path.join(data_dir, "training_rating.dat")
test_file = os.path.join(data_dir, "testing.dat")
output_file = os.path.join(data_dir, "result.csv")


'''
Tunable parameters
'''
D = 50             #number of factors
eta = 0.01         #learning rate
lambdaU = 0.1
lambdaV = 0.1
maxRating = 5
als_iterations = 10

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

# Build dictionaries for user-movie ratings
def get_dictionaries(userid_list, movieid_list, rating_list):
    number_users = max(userid_list)
    number_movies = max(movieid_list)
    rating_list = normalize_ratings(rating_list)
    userMovieDict  = dict()
    for i in range(0, len(userid_list)):
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
	    ratings[i][j] = (ratings[i][j] * (maxRating - 1) ) + 1
	    if (ratings[i][j] > 5):
                ratings[i][j] = 5
            if (ratings[i][j] < 1):
                ratings[i][j] = 1
    return ratings

# TODO: update using matrix
def loss(U, V, userMovieDict):
    loss = 0
    product = expit(U.T.dot(V))
    for i in range (0, product.shape[0]):
        for j in range(0, product.shape[1]):
            if (i+1) in userMovieDict and (j+1) in userMovieDict[i+1]:
                loss = loss + (userMovieDict[i+1][j+1] - product[i][j]) ** 2

    loss = loss * 1.0/2
    loss = loss +  (lambdaU * 1.0/2) * (LA.norm(U, 'fro') ** 2)
    loss = loss + (lambdaV * 1.0/2) * (LA.norm(V, 'fro') ** 2)
    return loss


# TODO: update using matrix
def RMSE(validDict, predicts):
    rmse = 0.0
    counter = 0
    for user in validDict:
	for movie in validDict[user]:
	    actualRating = (validDict[user][movie] * (maxRating - 1)) + 1
	    predictedRating = predicts[user - 1][movie - 1]
	    rmse = rmse + (actualRating-predictedRating)**2
            counter = counter + 1

    rmse = (rmse * 1.0/counter) ** 0.5
    return rmse

def ALS(U, V, ratings_matrix):

    # update U, latent vector U, fixed vector V
    for i in range(als_iterations):
        VTV = V.dot(V.T)
        lambdaU_matrix = np.eye(VTV.shape[0]) * lambdaU
        for u in xrange(U.shape[1]):
            U[u, :] = solve((VTV + lambdaU_matrix), ratings_matrix[u, :].T.dot(V.T)

    # update V
    for j in range (als_iterations):
        UTU = U.dot(U.T)
        lambdaV_matrix = np.eye(UTU.shape[0]) * lambdaV
        for v in xrange(V.shape[1]):
            V[v, :] = solve((UTU + lambdaV_matrix), ratings_matrix[:, v].T.dot(U.T))

    return U, V

def main():
    training_data, validation_data = split_training_data(training_file)
    training_userid_list, training_movieid_list, training_rating_list = preprocess_training_file(training_data)
    valid_userid_list, valid_movieid_list, valid_rating_list = preprocess_training_file(validation_data)

    userMovieDict, number_users, number_movies = get_dictionaries(training_userid_list, training_movieid_list, training_rating_list)
    valid_user_movie_dict, valid_number_users, valid_number_movies = get_dictionaries(valid_userid_list, valid_movieid_list, valid_rating_list)

    # TODO: replace 0 rating using mean ratings?
    ratings_matrix = coo_matrix((ratings_list, (training_userid_list, training_movieid_list)),
                    shape=(number_users, number_movies), dtype='float32')
    U = np.random.rand(D, number_users)
    V = np.random.rand(D, number_movies)

    for i in range (0, 65):
        U, V = ALS(U, V, ratings_matrix)

        predictions = U.T.dot(V)
        # TODO:
        lossValTrain = loss(U, V, userMovieDict)
        rmseTrain = RMSE (userMovieDict, predictions)
        lossValValid = loss(U, V, valid_user_movie_dict)
        rmseValid = RMSE(valid_user_movie_dict, predictions)
        print "Train Loss ", lossValTrain
        print "Train RMSE ", rmseTrain
        print "ValidSet Loss ", lossValValid
        print "ValidSet RMSE ", rmseValid
        print ""

    #TESTING CODE FOLLOWS
    #userid_list, movieid_list = preprocess_test_file(test_file)
    #U = pickle.load(open("U64", "rb"))
    #V = pickle.load(open("V64", "rb"))
    #ratings = U.T.dot(V)
    #ratings = nonnormalize_ratings(ratings)
    #for i in range(0, len(userid_list)):
    #    user = userid_list[i]
    #    movie = movieid_list[i]
    #    print ratings[user - 1][movie - 1]

main()
