#!/bin/sh
set -e -x

pids=""
for ch in 17 18 20 21 22 23
do
    ../convert2h5.py \
    --json mic-test.json \
    --chassis $ch \
    --output mic-test-CH${ch}.h5 \
    mic-test-CH${ch}-*.dat &
    pids="$pids $!"
done
wait $pids

pids=""
for ch in 17 18 20 21 22 23
do
    ../h52uff58.py --append \
    mic-test-CH${ch}.h5 \
    mic-test-CH${ch}.uff &
    pids="$pids $!"
done
wait $pids

cat mic-test-CH*.uff > mic-test.uff
