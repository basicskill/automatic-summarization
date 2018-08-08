#! /bin/bash

#cpuNumber=$(grep -c ^processor /proc/cpuinfo) # all cpu's
cpuNumber=3
check=1
workers=($(seq 1 $cpuNumber))

for ((i=0; i<$cpuNumber; i++)) do
    workers[$i]=0
done

./initialize.py

####
#calc=repikl1.py
#apply=apply_grads.py
noEpoch=1
data="./data2002repikl/"
####

fileCount=`ls data2002repikl/ | wc | awk '{print $1}'`

for ((i=0;i<$noEpoch;i++)) do
    br=0
    while [ $br -lt $fileCount ]; do
        for cpu in $(seq 1 $cpuNumber); do
            #echo kurac
            if [ ! -e /proc/${workers[$cpu]} ] && [ $br -ne $fileCount ]; then
                echo "Broj je: $br/$fileCount"
                taskset $cpu ./gradients.py $br $cpuNumber & 
                workers[$cpu]=$!
                echo "Worker $cpu started tree $br"
                ((br++))
            fi
        done
    done
done

echo "Kraj"
