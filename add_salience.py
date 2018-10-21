import os, sys
import tree
import pickle
import time


if len(sys.argv) != 4:
    print("""Usage: python add_salience.py 
        pickle_dir summeries_dir repickle_dir""")
    exit(1)

pickle_dir = sys.argv[1]
summeries_dir = sys.argv[2]
repickle_dir = sys.argv[3]

def write_saliences(claster, pickle_name):
    name = pickle_name.split('.')[0]
    loc = summeries_dir + '/' + name + '/'
    summs = [loc+x for x in os.listdir(loc)]
    for fajl in claster:
        for sentence in fajl:
            sentence.addSalience(summs, 0.5)
    return claster

pocetak = 19

for idx, claster_name in enumerate(sorted(os.listdir(pickle_dir))[pocetak:]):
    start = time.time()
    print("Writing saliences for: {} ({}/{})".format(claster_name, idx+1+pocetak, len(os.listdir(pickle_dir))))
    tt = pickle.load( open( pickle_dir+claster_name, 'rb' ) )
    tt_new = write_saliences(tt, claster_name)
    pickle.dump(tt_new, open(repickle_dir+claster_name, 'wb'))
    end = time.time()
    print("Time passed: {} s".format(int(end - start)))
