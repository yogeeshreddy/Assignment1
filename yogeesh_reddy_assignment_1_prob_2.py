import numpy as np

A=np.array([[1,0.67,0.33],[0.45,1,0.55],[0.67,0.33,1]])
b=np.array([2,2,2])

x=np.linalg.solve(A,b)

print("The solution obtained in my computer is",x,"whereas the solution obtained in first problem (by solving manually using only two floating point digits) is [1,1.1,0.95]")