#! /bin/bash

cpuNumber=$(grep -c ^processor /proc/cpuinfo)
check=1
workers=($(seq 1 $cpuNumber))
# echo $workers

####
calc=repikl1.py
apply=apply_grads.py
####

for cpu in $(seq 1 $cpuNumber); do
    taskset $cpu python $evaluate & workers[$cpu]=$!
done


while [[ $check -ne 0 ]]; do
    check=0
    for cpu in $(seq 1 $cpuNumber); do
        [ -e /proc/${workers[$cpu]} ] && check+=1 
    done
    # echo "ding";
    # sleep 1
done
python $apply $workers

echo "Kraj"
