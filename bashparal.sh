#! /bin/bash
start=$SECONDS
python initialize.py
taskset 1 python gradients.py 14 2 & p1=$!
#taskset 2 python gradients.py 14 2 & p2=$!
while [ ! -z "&(ps -p $p1 -o comm=)" ]
    do
        echo Kurac
    done
python apply_grads.py 2
duration=$(( $SECONDS - $start ))
echo $duration
