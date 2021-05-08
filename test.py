import numpy as np

def to_one_hot(a):
    b = np.zeros((a.size, 3))
    b[np.arange(a.size), a] = 1
    return b

if __name__ == '__main__':
    a = np.asarray([1,2,0,1,0,2])
    b = to_one_hot(a)
    c = np.argmax(b,axis=1)
    print(c)