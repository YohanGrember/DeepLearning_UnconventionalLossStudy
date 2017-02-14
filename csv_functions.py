# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:52:36 2017

@author: yohan
"""
import time
import csv

# Cette fonction permet de créer un csvwriter qui écrit à la fin du fichier csv "filename"
# si le fichier n'existe pas encore, il est créé
def init_csv(filename):
    
    csvfile = open(filename, 'a')
    csvwriter = csv.writer(csvfile, delimiter = ';', quotechar = '"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow([''])
    csvwriter.writerow([''])
    csvwriter.writerow(['New Training started at %g '%time.time()])
    csvwriter.writerow([''])
    return csvfile, csvwriter
  
def init_train_csv(filename):
    csvfile, csvwriter = init_csv(filename)
    csvwriter.writerow(['Iterations'] + ['Epochs'] + ['Training Risk'] + ['Computation Time'])
    return csvfile, csvwriter
    
def init_test_csv(filename):
    csvfile, csvwriter = init_csv(filename)
    csvwriter.writerow(['Iterations'] + ['Epochs'] + ['Testing Risk'] + ['Computation Time'])
    return csvfile, csvwriter
    
# Initialize two different csv files
train_csvfile, train_csvwriter = init_train_csv('train.csv')
test_csvfile, test_csvwriter = init_test_csv('test.csv')

# Adds the row to test.csv
test_csvwriter.writerow([50000] +[50] + [0.12] + [1200])

# Close the file at the end of the computations to write the last line
train_csvfile.close()
test_csvfile.close()

  
#cf, cw = init_csv('essai.csv')
#
#cw.writerow(['Nouvelle ligne 1'])
#cw.writerow(['Nouvelle ligne 2'])
#cf.close()