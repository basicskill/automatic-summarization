import sys, os
from stanfordnlp import read_xml

if len(sys.argv) != 3:
    print("Usage: python raw2java.py raw_folder java_folder")
    sys.exit()

raw = sys.argv[1]
java = sys.argv[2]

#os.rmdir(java)
#os.makedirs(java)

for claster in os.listdir(raw):
    os.makedirs(java + claster)
    print(claster)
    for fajl in os.listdir(raw+claster+'/'):
        try:
            txt = read_xml(raw+claster+'/'+fajl)
        except:
            print("Los fajl: {}".format(fajl))
            continue

        with open(java+claster+'/'+fajl, 'w') as f:
            f.write(txt)
