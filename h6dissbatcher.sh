#!/bin/bash

for i in {1..6}; do
    let "a = $i * 50 - 50"
    let "b = $a + 50"
    echo $a
    echo $b
    python h6dissbatch.py $a $b outeq &

done
wait


exit
