import numpy as np

def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    #Your code here
    #raise NotImplementedError

    s = np.linalg.norm(A+B)

    #print('A:',A)
    #print('B:',B)
    #print('s:',s)
    
    return s


X = np.random.random([2,2])
Y = np.random.random([2,2])
print(norm(X,Y))
    
