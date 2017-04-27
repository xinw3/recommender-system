import os
import time

# TODO: build matrix
data_dir = os.path.join('../', 'RSdata/')
file_name = 'training_rating.dat'
userid_list = []
movieid_list = []
ratings_list = []

def load_dataset(filename, separator):
    with open(filename, 'r') as training_data:
        count = 0
        for line in training_data:
            count += 1
            line = line.rstrip('\n').split(separator)[:3]
            if has_empty(line):
                continue
            else:
                #print line
                userid_list.append(line[0])
                movieid_list.append(line[1])
                ratings_list.append(line[2])
        print 'line numbers: ', count

def has_empty(line):
    for l in line:
        if not l.strip():
            return True

def main():
    training_file = os.path.join(data_dir, file_name)
    load_dataset(training_file, "::")
    #print has_empty(['', '  ', 'd'])
    print "len(user_id) = %d, len(movieid_list) = %d, len(ratings_list) = %d" % (len(userid_list), len(movieid_list), len(ratings_list))

start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
