import os
from scipy.sparse import coo_matrix
from sklearn import preprocessing
from numpy import linalg as LA
import numpy as np
import time

data_dir = os.path.join('../', 'RSdata/')
training_file = os.path.join(data_dir, "training_rating.dat")
test_file = os.path.join(data_dir, "testing.dat")
output_file = os.path.join(data_dir, "result.csv")


'''
Tunable parameters
'''
D = 50             #number of factors
eta = 0.0006    #learning rate
l = 0.1            #lambdaU, lambdaV

def preprocess_test_file():
    movieid_list = []
    userid_list = []
    with open(test_file, "r") as test_data:
        for line in test_data:
            elements = line.split(" ")
            movieid_list.append(int(elements[0]))
            userid_list.append(int(elements[1]))
        test_data.close()
    return movieid_list, userid_list

def preprocess_training_file():
    userid_list = []
    movieid_list = []
    rating_list = []
    with open(training_file, "r") as training_data:
        for line in training_data:
            elements = line.rstrip("\n").split("::")
            if has_empty(elements):
                continue
            else:
                userid_list.append(int(elements[0]))
                movieid_list.append(int(elements[1]))
                rating_list.append(int(elements[2]))
    # if has_empty(elements):
            #     continue
            # else:


    training_data.close()

    number_users = max(userid_list) + 1
    number_movies = max(movieid_list) + 1
    averageRating = sum (rating_list)/float(len(rating_list))

    userMovieDict  = dict()
    for i in range(0, len(userid_list)):
        user = userid_list[i]
        movie = movieid_list[i]
        rating = rating_list[i] - averageRating
        movieRatingsDict = dict()
        if user in userMovieDict:
            movieRatingsDict = userMovieDict[user]
        movieRatingsDict[movie] = rating
        userMovieDict[user] = movieRatingsDict
    return userMovieDict, number_users, number_movies, averageRating

def has_empty(elements):
    for e in elements:
        if not e.strip():
            return True
    return False

def loss(U, V, userMovieDict):
    loss = 0
    product = U.T.dot(V)

    for i in range (0, product.shape[0]):
        for j in range(0, product.shape[1]):
            if i in userMovieDict and j in userMovieDict[i]:
                loss = loss + (userMovieDict[i][j] - product[i][j]) ** 2

    loss = loss/2
    loss = loss +  (l/2) * (LA.norm(U, 'fro') ** 2)
    loss = loss + (l/2) * (LA.norm(V, 'fro') ** 2)

    return loss

def RMSE(true_ratings, predict_ratings):
    num_ratings = predict_ratings.shape[0];
    sum_squared_error = np.sum(np.square(predict_ratings - true_ratings))
    rmse = np.sqrt(np.divide(sum_squared_error, num_ratings - 1))
    return rmse

def ALS(U, V, userMovieDict):
    subtractionMatrix = np.ndarray(shape=(D,1))
    product = U.T.dot(V)

    for i in range (0, product.shape[0]):
        derivative = np.zeros((D, 1))
        for j in range(0, product.shape[1]):
            if i in userMovieDict and j in userMovieDict[i]:
                Vj =  np.reshape(V[:,j], (D, 1))
                derivative = derivative + (product[i][j] - userMovieDict[i][j]) * Vj
        Ui = np.reshape(U[:,i], (D, 1))
        derivative = derivative + (l * Ui)
        subtractionMatrix = np.hstack((subtractionMatrix , eta * derivative))

    subtractionMatrix  = np.delete(subtractionMatrix , 0, 1)
    U = U - subtractionMatrix

    subtractionMatrix = np.ndarray(shape=(D,1))
    product = U.T.dot(V)

    for j in range (0, product.shape[1]):
        derivative = np.zeros((D, 1))
        for i in range(0, product.shape[0]):
            if i in userMovieDict and j in userMovieDict[i]:
                Ui =  np.reshape(U[:,i], (D, 1))
                derivative = derivative + (product[i][j] - userMovieDict[i][j]) * Ui
        Vj = np.reshape(V[:,j], (D, 1))
        derivative = derivative + (l * Vj)
        subtractionMatrix = np.hstack((subtractionMatrix , eta * derivative))

    subtractionMatrix  = np.delete(subtractionMatrix , 0, 1)
    V = V - subtractionMatrix
    return U, V

def main():
    userMovieDict, number_users, number_movies, averageRating = preprocess_training_file()

    U = np.random.rand(D, number_users)
    V = np.random.rand(D, number_movies)

    lossVal = loss(U, V, userMovieDict)
    print lossVal
    for i in range (0, 100):
        U, V = ALS(U, V, userMovieDict)
        lossVal = loss(U, V, userMovieDict)
        print lossVal

    K = U.T.dot(V)

    query_movieid_list, query_userid_list = preprocess_test_file()
    output_fd = open(output_file, 'w')
    movieid_counter = 0
    for userid in query_userid_list:
        movieid = query_movieid_list[movieid_counter]
        movieid_counter = movieid_counter + 1
        if (userid >= number_users):
            output_fd.write(str(averageRating ))

        else:
            val = K[userid][movieid] + averageRating
            if (val < 1):
                val = 1
            elif (val > 5):
                val = 5
            output_fd.write(str(val ))
        output_fd.write("\n")
    output_fd.close()


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
