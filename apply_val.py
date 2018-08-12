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
        maploss = np.memmap("/tmp/pfedata/validation/tmp/" + str(i), dtype='float32', mode='r+', shape=())
        loss += maploss
    loss = loss/no_cpu
    with open("/tmp/pfedata/validation/losses", 'a') as f:
        np.savetxt(f, np.array([loss]))
"""    
    mapwp = np.memmap("/tmp/pfedata/weights/wp", dtype='float32', mode='r+', shape=(15, 8))
    Wp_reg[:] = mapwp[:]
    mapbp = np.memmap("/tmp/pfedata/weights/bp", dtype='float32', mode='r+', shape=(1, 8))
    bp_reg[:] = mapbp[:]
    mapwt = np.memmap("/tmp/pfedata/weights/wt", dtype='float32', mode='r+', shape=(16, 8))
    Wt_reg[:] = mapwt[:]
    mapbt = np.memmap("/tmp/pfedata/weights/bt", dtype='float32', mode='r+', shape=(1, 8))
    bt_reg[:] = mapbt[:]
    mapwr1 = np.memmap("/tmp/pfedata/weights/wr1", dtype='float32', mode='r+', shape=(8, 1))
    Wr1_reg[:] = mapwr1[:]
    mapwr2 = np.memmap("/tmp/pfedata/weights/wr2", dtype='float32', mode='r+', shape=(15, 1))
    Wr2_reg[:] = mapwr2[:]
    mapwr3 = np.memmap("/tmp/pfedata/weights/wr3", dtype='float32', mode='r+', shape=(14, 1))
    Wr3_reg[:] = mapwr3[:]
    mapbr = np.memmap("/tmp/pfedata/weights/br", dtype='float32', mode='r+', shape=(1, 1))
    br_reg[:] = mapbr[:]
"""







