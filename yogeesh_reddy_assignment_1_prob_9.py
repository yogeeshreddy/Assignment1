import numpy as np

A=np.array([[0.2,0.1,1.,1.,0.],[0.1,4.,-1.,1.,-1.],[1.,-1.,60.,0.,-2.],[1.,1.,0.,8.,4.],[0.,-1.,-2.,4.,700.]])
b=np.array([1.,2.,3.,4.,5.])
xreal=np.array([7.859713071,0.422926408,-0.073592239,-0.540643016,0.010626163])
x0=np.array([0.,0.,0.,0.,0.])

def norm(a,b):
    if len(a)==len(b):
        s=0
        for i in range(len(a)):
            s+=(a[i]-b[i])**2
        return s**0.5
    else:
        print("Error: The function 'norm' is defined only for vectors in same vector space.")


def Jacobi(A,b,tol,xreal):
    x=np.zeros(len(xreal))
    n=0
    while norm(x,xreal)>tol:
        y=x[:]
        x=np.zeros(len(xreal))
        for i in range(len(x)):
            for j in range(len(x)):
                x[i]+=(-A[i,j]*y[j])
            x[i]+=(b[i]+A[i,i]*y[i])
            x[i]=x[i]/A[i,i]
        n+=1
    print("The no.of iterations required using Jacobi method for required tolerance is",n)
    return x

def GaussSeidel(A0,b0,tol,xreal):
    x=np.zeros(len(xreal))
    n=0
    while norm(x,xreal)>tol:        
        for i in range(len(x)):
            x[i]=0
            for j in range(len(x)):
                if j!=i:
                    x[i]+=(-A0[i,j]*x[j])
                else:
                    continue
            x[i]+=(b0[i])
            x[i]=x[i]/A0[i,i]
        n+=1
    print("No. of iterations required using Gauss-Seidel method for required tolerance is",n)
    return x

def Relaxation(A0,b0,omega,tol,xreal):
    x=np.zeros(len(xreal))
    n=0
    while norm(x,xreal)>tol:
        for i in range(len(x)):
            x[i]=(1-omega)*x[i]+(omega/A0[i,i])*b[i]
            for j in range(len(x)):
                if j!=i:
                    x[i]+=(omega/A0[i,i])*(-A0[i,j]*x[j])
                else:
                    continue
        n+=1
    print("No. of iterations required using Relaxation method for required tolerance is",n)
    return x

def ConjGrad(A,b,tol,xreal):
    x=np.zeros(len(xreal))
    r=b-np.dot(A,x)
    print(r)
    p=r[:]
    n=0
    while n<10:
        alpha=(np.dot(r,r))/(np.dot(r,np.dot(A,p)))
        x+=alpha*p
        r2=r[:]
        r+=(-alpha*np.dot(A,p))
        beta=(np.dot(r,r))/(np.dot(r2,r2))
        p=beta*p+r
        n+=1
    print("No. of iterations required using Conjugate Gradient method for required tolerance is",n)
    return x

A2=np.matrix([[.2,.1,1,1,0],[.1,4,-1,1,-1],[1,-1,60,0,-2],[1,1,0,8,4],[0,-1,-2,4,700]])
b2=np.transpose(np.matrix([1,2,3,4,5]))
xreal2=np.transpose((np.matrix([7.859713071,.422926408,-.073592239,-.540643016,.010626163])))

def ConjGrad2(A,b,tol,xreal):
    x=np.transpose(np.matrix(np.zeros(len(xreal))))
    r=b-A*x
    print(r)
    p=r[:]
    n=0
    while n<10:
        alpha=(np.transpose(r)*r)[0,0]/(np.transpose(p)*A*p)[0,0]
        x=x[:]+(p*alpha)[:]
        r=r-A*p*alpha
        beta=(np.transpose(r)*r)[0,0]/(np.transpose(r+A*p*alpha)*(r+A*p*alpha))[0,0]
        p=r+p*beta
        n+=1
    print("No. of iterations required using Conjugate Gradient method for required tolerance is",n)
    return x




    
print(Jacobi(A,b,0.01,xreal))
print("***********************************")
print(GaussSeidel(A,b,0.01,xreal))
print("***********************************")
print(Relaxation(A,b,1.25,0.01,xreal))
print("***********************************")
print(ConjGrad(A,b,0.01,xreal))
print("***********************************")
print(ConjGrad2(A2,b2,0.01,xreal2))
