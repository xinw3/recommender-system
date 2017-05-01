import numpy as np
# check if the string is empty
# print not "    "
#
# print not "    ".strip()
lambdaU_out = "lambdaU.npy"
x = np.arange(10)
np.save(lambdaU_out, x)
y = np.load(lambdaU_out)
print y
