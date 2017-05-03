import os
from scipy.sparse import coo_matrix
from sklearn import preprocessing
from numpy import linalg as LA
import numpy as np
import time
from scipy.special import expit
import random
import pickle

data_dir = os.path.join('../', 'RSdata/')
training_file = os.path.join(data_dir, "training_rating.dat")
test_file = os.path.join(data_dir, "testing.dat")
output_file = os.path.join(data_dir, "result.csv")


'''
Tunable parameters
'''
D = 50             #number of factors
eta = 0.0001         #learning rate
lambdaU = 0.001
lambdaV = 0.001
maxRating = 5

def subtractAverage(rating_list, averageRating):
    rating_list[:] = [x - averageRating for x in rating_list]
    return rating_list

def averageCalc(rating_list):
    averageRating = sum (rating_list)/float(len(rating_list))
    return averageRating

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
    pickle.dump(validation_indices, open("validation_indices_linear", "wb"))
    #validation_indices = pickle.load(open("validation_indices_linear", "rb"))
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

def RMSE(validDict, predicts, averageRating):
    rmse = 0.0
    counter = 0
    for user in validDict:
    	for movie in validDict[user]:
    	    actualRating = validDict[user][movie] + averageRating
    	    predictedRating = predicts[user - 1][movie - 1] + averageRating
	    if (predictedRating < 1):
                predictedRating = 1.0
            elif (predictedRating > 5):
		predictedRating = 5.0
    	    rmse = rmse + (actualRating-predictedRating)**2
            counter = counter + 1

    rmse = (rmse * 1.0/counter) ** 0.5
    return rmse

def ALS(U, V, userMovieDict):
    subtractionMatrix = np.ndarray(shape=(D,1))
    product = U.T.dot(V)

    for i in range (0, product.shape[0]):
        derivative = np.zeros((D, 1))
        for j in range(0, product.shape[1]):
            if i+1 in userMovieDict and j+1 in userMovieDict[i+1]:
                Vj =  np.reshape(V[:,j], (D, 1))
                derivative = derivative + (product[i][j] - userMovieDict[i+1][j+1]) * Vj
        Ui = np.reshape(U[:,i], (D, 1))
        derivative = derivative + (lambdaU * Ui)
        subtractionMatrix = np.hstack((subtractionMatrix , derivative))

    subtractionMatrix  = np.delete(subtractionMatrix , 0, 1)
    U = U - (eta * subtractionMatrix)

    subtractionMatrix = np.ndarray(shape=(D,1))
    product = U.T.dot(V)

    for j in range (0, product.shape[1]):
        derivative = np.zeros((D, 1))
        for i in range(0, product.shape[0]):
            if i+1  in userMovieDict and j+1 in userMovieDict[i+1]:
                Ui =  np.reshape(U[:,i], (D, 1))
                derivative = derivative + (product[i][j] - userMovieDict[i+1][j+1]) * Ui
        Vj = np.reshape(V[:,j], (D, 1))
        derivative = derivative + (lambdaV * Vj)
        subtractionMatrix = np.hstack((subtractionMatrix , derivative))

    subtractionMatrix  = np.delete(subtractionMatrix , 0, 1)
    V = V - (eta * subtractionMatrix)
    return U, V

def main():
    training_data, validation_data = split_training_data(training_file)
    training_userid_list, training_movieid_list, training_rating_list = preprocess_training_file(training_data)
    valid_userid_list, valid_movieid_list, valid_rating_list = preprocess_training_file(validation_data)
    
    averageRatingTrain = averageCalc(training_rating_list)
    
    training_rating_list = subtractAverage(training_rating_list, averageRatingTrain)
    valid_rating_list = subtractAverage(valid_rating_list, averageRatingTrain) 

    userMovieDict, number_users, number_movies = get_dictionaries(training_userid_list, training_movieid_list, training_rating_list)
    valid_user_movie_dict, valid_number_users, valid_number_movies = get_dictionaries(valid_userid_list, valid_movieid_list, valid_rating_list)

    U = np.random.rand(D, number_users)
    V = np.random.rand(D, number_movies)
    #U = pickle.load(open("U95", "rb"))
    #V = pickle.load(open("V95", "rb"))

    for i in range (0, 65):
        U, V = ALS(U, V, userMovieDict)
        pickle.dump(U, open("LU"+str(i), "wb"))
        pickle.dump(V, open("LV"+str(i), "wb"))
        lossValTrain = loss(U, V, userMovieDict)
        predictions = U.T.dot(V)
        rmseTrain = RMSE (userMovieDict, predictions, averageRatingTrain)
        lossValValid = loss(U, V, valid_user_movie_dict)
        rmseValid = RMSE(valid_user_movie_dict, predictions, averageRatingTrain)
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
