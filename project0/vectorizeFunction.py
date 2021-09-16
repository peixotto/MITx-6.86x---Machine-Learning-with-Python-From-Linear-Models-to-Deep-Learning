import numpy as np

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.

    f(x,y) = { x.y, if x<= y    }
             { x/y, else        }
    """
    #Your code here
    #raise NotImplementedError

    if x<=y:
        f = x*y
    else:
            f = x/y
    
    return f

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    #raise NotImplementedError

    vf = np.vectorize(scalar_function)
    return vf(x,y)

print(scalar_function(5,2), type(scalar_function(5,2)))
print(scalar_function(1,2), type(scalar_function(1,2)))

print(vector_function(5,2), type(vector_function(5,2)))
print(vector_function(1,2), type(vector_function(1,2)))