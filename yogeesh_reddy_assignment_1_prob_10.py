import numpy as np
from numpy import linalg as npl

A=np.array([[5,-2],[-2,8]])

def QReig(A):
    for i in range(23):
        B=npl.qr(A)[:]
        A=np.dot(B[1],B[0])
    return [A,B]

# =============================================================================
# print("From QR decomposition, after 23 iterations, the eigenvalues are", QReig(A)[0,0],"and",QReig(A)[1,1])
# print("From linalg.eigh we the the eigenvalues to be", npl.eigh(A)[0][1],"and",npl.eigh(A)[0][0])
# =============================================================================

print(QReig(A))