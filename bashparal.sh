#! /bin/bash

#cpuNumber=$(grep -c ^processor /proc/cpuinfo) # all cpu's
cpuNumber=$1
((cpuNumber--))

workers=($(seq 0 $cpuNumber))

for ((i=1; i<=$cpuNumber; i++)) do
    workers[$i]=0
done

python3 initialize.py

####
#calc=repikl1.py
#apply=apply_grads.py
noEpoch=10
data="/tmp/pfedata/demo/training/"
valid="/tmp/pfedata/demo/validation/"
####

fileCount=`ls $data | wc | awk '{print $1}'`
validCount=`ls $valid | wc | awk '{print $1}'`

for ((i=0;i<$noEpoch;i++)) do
    echo "Start of Epoch $i"

    # Gradients

    job=0
    while [ $job -lt $fileCount ]; do
        start=$SECONDS
        cpu=0
        cpuNeed=$((fileCount-job > cpuNumber ? cpuNumber : fileCount-job))
        while [ $cpu -le $cpuNeed ]; do
            taskset -c $cpu python3 gradients.py $job $cpuNumber & 
            workers[$cpu]=$!
            echo -e " \t Worker $cpu started tree $job/$((fileCount-1))"
            ((job++))
            ((cpu++))
        done

        echo -e " \t Waiting for workers to finish..."
        
        finish=0
        cpu=1
        while [ $finish -le $cpuNumber ]; do
            if [ ! -e /proc/${workers[$cpu]} ]; then
                let "cpu=cpu%cpuNumber+1"
                ((finish++))
            fi
        done

        echo -e " \t Applying grads..."
        python3 apply_grads.py $cpuNumber
        duration=$(( $SECONDS - $start ))
        echo $duration
    done

    # Validaton 

    job=0
    while [ $job -lt $validCount ]; do
        cpu=0
        cpuNeed=$((validCount-job > cpuNumber ? cpuNumber : validCount-job))
        while [ $cpu -le $cpuNeed ]; do
            taskset -c $cpu python3 validate.py $job $cpuNumber & 
            workers[$cpu]=$!
            echo -e " \t Worker $cpu validating tree $job/$((validCount-1))"
            ((job++))
            ((cpu++))
        done

        echo -e " \t Waiting for workers to finish..."
        
        finish=0
        cpu=1
        while [ $finish -le $cpuNumber ]; do
            if [ ! -e /proc/${workers[$cpu]} ]; then
                let "cpu=cpu%cpuNumber+1"
                ((finish++))
            fi
        done

        echo -e " \t Applying validation..."
        python3 apply_val.py $cpuNumber
    done
    mkdir /tmp/pfedata/validation/$i
    cp /tmp/pfedata/weights/* /tmp/pfedata/validation/$i/
done


echo "Kraj"
