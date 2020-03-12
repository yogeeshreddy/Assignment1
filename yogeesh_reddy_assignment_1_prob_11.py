import numpy as np

A=np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
       
def Power(A):
    x=np.zeros(len(A))
    z=x[:]
    x[0]=1
    l=0
    u=0
    while abs(l-np.amax(np.linalg.eigh(A)[0]))>0.01:
        y=x[:]
        x=np.dot(A,x)[:]
        l=(np.dot(x,z))/(np.dot(y,z))
        u+=1
    return [l,x,u]

print("The dominant eigenvalue and eigenvector obtained using Power method is",Power(A)[0],"and",Power(A)[1]/Power(A)[1][1],"respectively. The number of iterations required to achieve the required tolerance is",Power(A)[2])