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
noEpoch=4
data="./data2002repikl/"
####

fileCount=`ls data2002repikl/ | wc | awk '{print $1}'`
sleep 10000

for ((i=0;i<$noEpoch;i++)) do
    br=0
    while [ $br -le $fileCount ]; do
        for cpu in $(seq 1 $cpuNumber); do
            #echo kurac
            if [ ! -e /proc/${workers[$cpu]} ]; then
                taskset $cpu ./gradients.py $br $cpuNumber & ((br++)) & workers[$cpu]=$!
                echo "Worker $cpu started tree $br"
            fi
        done
    done
done


while [[ $check -ne 0 ]]; do
    check=0
    for cpu in $(seq 1 $cpuNumber); do
        [ -e /proc/${workers[$cpu]} ] && check+=1 
    done
    # echo "ding";
    # sleep 1
done
python $apply $cpuNumber

echo "Kraj"
