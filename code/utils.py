import os
import time
from scipy.sparse import coo_matrix

data_dir = os.path.join('../', 'RSdata/')
file_name = 'training_rating.dat'


def load_dataset(filename, separator):
    userid_list = []
    movieid_list = []
    ratings_list = []
    with open(filename, 'r') as training_data:
        count = 0
        for line in training_data:
            # count += 1
            line = line.rstrip('\n').split(separator)[:3]
            if has_empty(line):
                continue
            else:
                # print line
                userid_list.append(int(line[0]))
                movieid_list.append(int(line[1]))
                ratings_list.append(int(line[2]))
        # print 'line numbers: ', count
    num_users = max(userid_list) + 1        # TODO: seems there is no userid = 0 and movie_id = 0
    num_movies = max(movieid_list) + 1
    training_data.close()
    matrix_coo = coo_matrix((ratings_list, (userid_list, movieid_list)),
                    shape=(num_users, num_movies), dtype='float32')
    return matrix_coo, userid_list, movieid_list, ratings_list

def has_empty(line):
    for l in line:
        if not l.strip():
            return True
    return False

def RMSE(true_ratings, predict_ratings):
    num_test = predict_ratings.shape[0];
    squared_sum_error = np.sum(np.square(predict_ratings - true_ratings))
    rmse = np.sqrt(np.divide(squared_sum_error, num_test - 1))
    return rmse

def main():
    training_file = os.path.join(data_dir, file_name)
    (matrix_coo, userid_list, movieid_list, ratings_list) = load_dataset(training_file, "::")
    #print has_empty(['', '  ', 'd'])
    print "len(user_id) = %d, len(movieid_list) = %d, len(ratings_list) = %d" % (len(userid_list), len(movieid_list), len(ratings_list))
    print 'matrix_coo shape: ', matrix_coo.shape
    matrix_arr = matrix_coo.toarray();

start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
