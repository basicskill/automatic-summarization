#! /usr/bin/python

import time
if __name__ == "__main__":
    start = time.time()
    import numpy as np


    Wr1_reg = np.random.normal(0.0, 0.1, [8, 1])
    Wr2_reg = np.random.normal(0.0, 0.1, [15, 1])
    Wr3_reg = np.random.normal(0.0, 0.1, [14, 1])
    br_reg = np.random.normal(0.0, 0.1, [1, 1])

    Wt_reg = np.random.normal(0.0, 0.1, [16, 8])
    bt_reg = np.random.normal(0.0, 0.1, [1, 8])

    Wp_reg = np.random.normal(0.0, 0.1, [15, 8])
    bp_reg = np.random.normal(0.0, 0.1, [1, 8])

    mapwp = np.memmap("./weights/wp", dtype='float32', mode='w+', shape=(15, 8))
    mapwp[:] = Wp_reg[:]
    mapbp = np.memmap("./weights/bp", dtype='float32', mode='w+', shape=(1, 8))
    mapbp[:] = bp_reg[:]
    mapwt = np.memmap("./weights/wt", dtype='float32', mode='w+', shape=(16, 8))
    mapwt[:] = Wt_reg[:]
    mapbt = np.memmap("./weights/bt", dtype='float32', mode='w+', shape=(1, 8))
    mapbt[:] = bt_reg[:]
    mapwr1 = np.memmap("./weights/wr1", dtype='float32', mode='w+', shape=(8, 1))
    mapwr1[:] = Wr1_reg[:]
    mapwr2 = np.memmap("./weights/wr2", dtype='float32', mode='w+', shape=(15, 1))
    mapwr2[:] = Wr2_reg[:]
    mapwr3 = np.memmap("./weights/wr3", dtype='float32', mode='w+', shape=(14, 1))
    mapwr3[:] = Wr3_reg[:]
    mapbr = np.memmap("./weights/br", dtype='float32', mode='w+', shape=(1, 1))
    mapbr[:] = br_reg[:]

