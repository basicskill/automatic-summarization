#! /bin/bash

#cpuNumber=$(grep -c ^processor /proc/cpuinfo) # all cpu's
cpuNumber=$1

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
valid="./valid_set"
####

fileCount=`ls $data | wc | awk '{print $1}'`
validCount=`ls $valid | wc | awk '{print $1}'`

for ((i=0;i<$noEpoch;i++)) do
    echo "Start of Epoch $i"

    # Gradients

    job=0
    while [ $job -lt $fileCount ]; do
        start=$SECONDS
        cpu=1
        cpuNeed=$((fileCount-job > cpuNumber ? cpuNumber : fileCount-job))
        while [ $cpu -le $cpuNeed ]; do
            taskset $cpu ./gradients.py $job $cpuNumber & 
            workers[$cpu]=$!
            echo "Worker $cpu started tree $job/$((fileCount-1))"
            ((job++))
            ((cpu++))
        done

        echo "Waiting for workers to finish..."
        
        finish=0
        cpu=1
        while [ $finish -le $cpuNumber ]; do
            if [ ! -e /proc/${workers[$cpu]} ]; then
                let "cpu=cpu%cpuNumber+1"
                ((finish++))
            fi
        done

        echo "Applying grads..."
        ./apply_grads.py $cpuNumber
        duration=$(( $SECONDS - $start ))
        echo $duration
    done

    # Validaton 

    job=0
    while [ $job -lt $validCount ]; do
        cpu=1
        cpuNeed=$((validCount-job > cpuNumber ? cpuNumber : validCount-job))
        while [ $cpu -le $cpuNeed ]; do
            taskset $cpu ./gradients.py $job $cpuNumber & 
            workers[$cpu]=$!
            echo "Worker $cpu validating tree $job/$((validCount-1))"
            ((job++))
            ((cpu++))
        done

        echo "Waiting for workers to finish..."
        
        finish=0
        cpu=1
        while [ $finish -le $cpuNumber ]; do
            if [ ! -e /proc/${workers[$cpu]} ]; then
                let "cpu=cpu%cpuNumber+1"
                ((finish++))
            fi
        done

        echo "Applying validation..."
        ./apply_val.py $cpuNumber
    done
done


echo "Kraj"
