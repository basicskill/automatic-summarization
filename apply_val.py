#! /usr/bin/python

if __name__ == "__main__":
    import sys
    import numpy as np
    import pickle
    import os
    args = sys.argv
    no_cpu = int(args[1])

    loss = 0
    diffabs = 0
    diffsqr = 0

    for i in range(no_cpu):
        maploss = np.memmap("./validation/tmp/l" + str(i), dtype='float32', mode='r+', shape=())
        loss += maploss
    loss = loss/no_cpu

    for i in range(no_cpu):
        maploss = np.memmap("./validation/tmp/abs" + str(i), dtype='float32', mode='r+', shape=())
        diffabs += maploss
    diffabs = diffabs/no_cpu
    
    for i in range(no_cpu):
        maploss = np.memmap("./validation/tmp/sqr" + str(i), dtype='float32', mode='r+', shape=())
        diffsqr += maploss
    diffsqr = diffsqr/no_cpu
 
    with open("./validation/losses", 'a') as f:
        np.savetxt(f, np.array([loss]))
    
    with open("./validation/diffabs", 'a') as f:
        np.savetxt(f, np.array([diffabs]))

    with open("./validation/diffsqr", 'a') as f:
        np.savetxt(f, np.array([diffsqr]))








