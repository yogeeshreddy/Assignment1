import numpy as np
import time

A=np.matrix([[0,1,1],[0,1,0],[1,1,0],[0,1,0],[1,0,1]])

def SVD(A):
    A1=np.dot(A,np.transpose(A))
    A2=np.dot(np.transpose(A),A)
    v1=np.flip(np.transpose(np.flip(np.transpose(np.linalg.eigh(A1)[1]))),0)
    v2=np.flip(np.transpose(np.linalg.eigh(A2)[1]))
    if np.shape(A1)[0]>=np.shape(A2)[0]:
        S=np.flip(np.diag(np.sqrt(np.linalg.eigh(A2)[0])))
        for i in range(np.shape(A1[0])[1]-np.shape(A2[0])[1]):
            S=np.append(S, [np.zeros(np.shape(S[0])[0])], axis=0)
    else:
        S=np.flip(np.diag(np.sqrt(np.linalg.eigh(A1)[0])))
        for i in range(np.shape(A2[0])[1]-np.shape(A1[0])[1]):
            S=np.append(S, [np.zeros(np.shape(S[0])[0])], axis=0)
    return[v1,S,v2]


print(SVD(A))
print()
print("The above matrices when multiplied give the original matrix. It is the code for SVD that I wrote")
print()
print("*************************************")
print()
print("The below is the decomposition obtained using numpy.linalg.svd")
print()
print(np.linalg.svd(A))

print("Now we'll compare the time taken for the two methods. Since the time measured by the process varies slightly every time, I'll average over 100000 iterations")

start = time.time()
for i in range(100000):
    SVD(A)
end = time.time()
print("The average time taken by my code is",(end - start)/100000,"seconds")

start2 = time.time()
for i in range(100000):
    np.linalg.svd(A)
end2 = time.time()
print("The average time taken by inbuilt function is",(end2 - start2)/100000,"seconds")

# =============================================================================
# In my computer the result for time taken was as follows:
#     
# The average time taken by my code is 7.918940305709839e-05 seconds
# The average time taken by inbuilt function is 1.7589986324310302e-05 seconds
# =============================================================================


# =============================================================================
# The matrices obtained using both the functions are slightly different. If the matrix is A=USV, then the last two columns 
#of U are not unique because of zero eigenvalue. Any two orthonormal vectors in the eigenspace of eigenvalue with be fine.
#The third column of U and the third row of V have signs opposite to that of the inbuilt function, but they cancel out to give
# the correct matrix after multiplication.
# =============================================================================
