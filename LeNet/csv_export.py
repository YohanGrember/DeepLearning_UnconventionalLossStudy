import os
import errno
import csv
import time


def init_csv(filename):
    csvfile = open(filename, 'a')
    csvwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow([''])
    csvwriter.writerow([''])
    csvwriter.writerow(['New Training started at %g ' % time.time()])
    csvwriter.writerow([''])
    return csvfile, csvwriter


def init_train_csv(filename):
    csvfile, csvwriter = init_csv(filename)
    csvwriter.writerow(['Loss Function'] + ['Iterations'] + ['Epochs'] + ['Training Accuracy'] + ['Computation Time'])
    csvfile.close()


#    return csvfile, csvwriter

def init_test_csv(filename):
    csvfile, csvwriter = init_csv(filename)
    csvwriter.writerow(['Loss Function'] + ['Iterations'] + ['Epochs'] + ['Testing Accuracy'] + ['Computation Time'])
    csvfile.close()


#    return csvfile, csvwriter

# Open a csv file, write a row at the end of it, and close it
def csv_writerow(csv_file, row):
    with open(csv_file, 'a') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)


# This function creates a path if it doesn't already exist
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
          raise