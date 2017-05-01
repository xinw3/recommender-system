import numpy as np
import os.path
# check if the string is empty
# print not "    "
#
# print not "    ".strip()
'''
Test save and load a file
'''
lambdaU_out = "lambdaU.npy"
x = np.arange(10)
np.save(lambdaU_out, x)
if os.path.isfile(lambdaU_out):
    print 'file exists'
    y = np.load(lambdaU_out)
else:
    print 'file does not exist'
    y = np.zeros(5)
print y
