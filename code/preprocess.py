import os
import time
from scipy.sparse import coo_matrix

# TODO: build matrix
data_dir = os.path.join('../', 'RSdata/')
file_name = 'training_rating.dat'


def load_dataset(filename, separator):
    with open(filename, 'r') as training_data:
        count = 0
        userid_list = []
        movieid_list = []
        ratings_list = []
        for line in training_data:
            count += 1
            line = line.rstrip('\n').split(separator)[:3]
            if has_empty(line):
                continue
            else:
                # print line
                userid_list.append(int(line[0]))
                movieid_list.append(int(line[1]))
                ratings_list.append(int(line[2]))
        num_users = max(userid_list) + 1
        num_movies = max(movieid_list) + 1
        matrix_coo = coo_matrix((ratings_list, (userid_list, movieid_list)),
    					shape=(num_users, num_movies), dtype='float32')
        print 'line numbers: ', count
        return matrix_coo, userid_list, movieid_list, ratings_list

def has_empty(line):
    for l in line:
        if not l.strip():
            return True

def main():
    training_file = os.path.join(data_dir, file_name)
    (matrix_coo, userid_list, movieid_list, ratings_list) = load_dataset(training_file, "::")
    #print has_empty(['', '  ', 'd'])
    print "len(user_id) = %d, len(movieid_list) = %d, len(ratings_list) = %d" % (len(userid_list), len(movieid_list), len(ratings_list))
    print 'matrix_coo shape: ', matrix_coo.shape

start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
