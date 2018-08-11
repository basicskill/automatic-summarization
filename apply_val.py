#! /usr/bin/python

if __name__ == "__main__":
    import sys
    import numpy as np
    import pickle
    import os
    args = sys.argv
    no_cpu = int(args[1])

    loss = 0

    for i in range(no_cpu):
        maploss = np.memmap("./validation/tmp/" + str(i), dtype='float32', mode='r+', shape=())
        loss += maploss
    loss = loss/no_cpu
    with open("./validation/losses", 'a') as f:
        np.savetxt(f, np.array([loss]))







