import os
from scipy.sparse import coo_matrix
from sklearn import preprocessing
from numpy import linalg as LA
import numpy as np
import time
from scipy.special import expit
from sklearn.model_selection import train_test_split
import random

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

def preprocess_test_file(test_file):
    movieid_list = []
    userid_list = []
    with open(test_file, "r") as test_data:
        for line in test_data:
            elements = line.split(" ")
            movieid_list.append(int(elements[0]))
            userid_list.append(int(elements[1]))
        test_data.close()
    return movieid_list, userid_list

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

def RMSE(true_ratings, predict_ratings):
    num_ratings = predict_ratings.shape[0] * predict_ratings.shape[1];
    sum_squared_error = np.sum(np.square(predict_ratings - true_ratings))
    rmse = np.sqrt(np.divide(sum_squared_error, num_ratings))
    return rmse

def ALS(U, V, userMovieDict):
    print "Calculating New U"
    product = U.T.dot(V)
    product_derivative = np.multiply(expit(product), 1 - expit(product))    # g(1-g)
    for i in range (0, product.shape[0]):
        for j in range(0, product.shape[1]):
            if (i+1) in userMovieDict and (j+1) in userMovieDict[i+1]:
		product[i][j] = product[i][j] - userMovieDict[i+1][j+1]
	    else:
		product[i][j] = 0

    derivative_matrix = np.multiply(product, product_derivative)
   
    subtractionMatrix = np.ndarray(shape=(D,1)) 
    for i in range(0, derivative_matrix.shape[0]):
	summation = np.zeros((D,1))
        for j in range(0, derivative_matrix.shape[1]):
	    Vj =  np.reshape(V[:,j], (D, 1))
	    Vj = Vj * derivative_matrix[i][j]
            summation = summation + Vj
        subtractionMatrix = np.hstack(( subtractionMatrix, summation))
    subtractionMatrix  = np.delete(subtractionMatrix , 0, 1)
    subtractionMatrix = subtractionMatrix + (lambdaU * U)
    U = U - (eta *subtractionMatrix)
    
    print "Calculating New V"
    product = U.T.dot(V)
    product_derivative = np.multiply(expit(product), 1 - expit(product))    # g(1-g)
    for i in range (0, product.shape[0]):
        for j in range(0, product.shape[1]):
            if (i+1) in userMovieDict and (j+1) in userMovieDict[i+1]:
		product[i][j] = product[i][j] - userMovieDict[i+1][j+1]
	    else:
		product[i][j] = 0

    derivative_matrix = np.multiply(product, product_derivative)
    
    subtractionMatrix = np.ndarray(shape=(D,1))
    for j in range (0, derivative_matrix.shape[1]):
        summation = np.zeros((D, 1))
        for i in range(0, derivative_matrix.shape[0]):
            Ui =  np.reshape(U[:,i], (D, 1))
            Ui = Ui * derivative_matrix[i][j]
            summation = summation + Ui
        subtractionMatrix = np.hstack((subtractionMatrix , summation))
    subtractionMatrix  = np.delete(subtractionMatrix , 0, 1)
    subtractionMatrix  = subtractionMatrix + (lambdaV * V)
    V = V - (eta * subtractionMatrix)
    
    print "ALS Round done"
    return U, V

def main():
    training_data, validation_data = split_training_data(training_file)
    training_userid_list, training_movieid_list, training_rating_list = preprocess_training_file(training_data)
    valid_userid_list, valid_movieid_list, valid_rating_list = preprocess_training_file(validation_data)

    userMovieDict, number_users, number_movies = get_dictionaries(training_userid_list, training_movieid_list, training_rating_list)
    valid_user_movie_dict, valid_number_users, valid_number_movies = get_dictionaries(valid_userid_list, valid_movieid_list, valid_rating_list)

    U = np.random.rand(D, number_users)
    V = np.random.rand(D, number_movies)

    lossVal = loss(U, V, userMovieDict)
    print lossVal
    for i in range (0, 100):
        U, V = ALS(U, V, userMovieDict)
        lossVal = loss(U, V, userMovieDict)
        print lossVal
    # TODO: Quicken ALS and Loss further ? Bettwe way to compose Indicator matrix?
    # TODO: Do we need to scale back?
    R_predict = U.T.dot(V)
    valid_matrix_coo = coo_matrix((valid_rating_list, (valid_userid_list, valid_movieid_list)),
                    shape=(valid_number_users, valid_number_movies), dtype='float32')

    # compute rmse using validation set
    # TODO: RMSE is not supposed to use in this way
    movieid_counter = 0
    for userid in valid_userid_list:
        movieid = valid_movieid_list[movieid_counter]
        rmse = RMSE(valid_matrix_coo[userid][movieid], R_predict[userid][movieid])
        print 'RMSE', rmse

    # TODO: write to result file
    # test_movieid_list, test_userid_list = preprocess_test_file(test_file)
    # output_fd = open(output_file, 'w')
    # movieid_counter = 0
    # for userid in test_userid_list:
    #     movieid = test_movieid_list[movieid_counter]
    #     movieid_counter = movieid_counter + 1
    #     val = R_predict[userid][movieid]
    #     output_fd.write(str(val))
    #     output_fd.write("\n")
    # output_fd.close()

start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
