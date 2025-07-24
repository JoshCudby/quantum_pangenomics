#!/bin/bash
cat ./mqlib.ga.000*/sim.out | grep 'Node count:' | awk '
BEGIN { 
    sum = 0; count = 0; max = 0; min = 1000 
} 
{
    print $3
    sum = sum + $3; 
    count = count + 1; 
    if ($3 <= min) { 
        min = $3
    }; 
    if ($3 >= max) {
        max = $3 
    }  
} 
END { 
    print sum; print count; print min; print max 
}'