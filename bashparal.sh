#! /bin/bash

#cpuNumber=$(grep -c ^processor /proc/cpuinfo) # all cpu's
cpuNumber=2

workers=($(seq 0 $cpuNumber))

for ((i=1; i<=$cpuNumber; i++)) do
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
((fileCount--))

for ((i=0;i<$noEpoch;i++)) do
    job=0
    cpu=0
    while [ $job -le $fileCount ]; do
        let "cpu=cpu%cpuNumber+1"
        if [ ! -e /proc/${workers[$cpu]} ]; then
            taskset $cpu ./gradients.py $job $cpuNumber & 
            workers[$cpu]=$!
            echo "Worker $cpu started tree $job/$fileCount @ ${workers[$cpu]}"
            ((job++))
        fi
    done

    echo "Waiting for workers to finish..."
    
    finish=0
    cpu=0
    while [ $finish -le $cpuNumber ]; do
        if [ ! -e /proc/${workers[$cpu]} ]; then
            let "cpu=cpu%cpuNumber+1"
            ((finish++))
        fi
    done

    echo "Applying grads..."
    ./apply_grads.py $cpuNumber

done



echo "Kraj"
