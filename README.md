# Interim Quartz data handling scripts

## Present limitations

Scripts work with data files recorded by a single chassis in a single acquisition.
They will __not__ combine/align data from multiple chassis.

## Pre-requisites

- scipy
- h5py
- matplotlib (display scripts only)

## Process

First combine raw sample files with calibration and scaling
into an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file.
`.dat` files must be provided in time order (by default encoded in filename).

```
./convert2h5.py \
 --calib calibration/20240326/sn1003-cal.txt \
 --scale end2end-scale.txt \
 --freq 50000 \
 -o EVG-50k-20240322-201504-full.h5 \
 EVG-50k-20240322-201504.dat \
 EVG-50k-20240322-202209.dat
```

Optional software [downsampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html#scipy.signal.decimate) with a FIR filter.

```
./downsamph5.py \
 --div 2 \
 EVG-50k-20240322-201504-full.h5 \
 EVG-25k-20240322-201504-full.h5
```

Export channels 1 through 13 as a [UFF](https://www.ceas3.uc.edu/sdrluff/all_files.php) [58b](https://www.svibs.com/resources/ARTeMIS_Modal_Help/UFF%20Data%20Set%20Number%2058.html) file.

```
./h52uff58.py \
 -M 0x1fff \
 EVG-25k-20240322-201504-full.h5 \
 EVG-25k-20240322-201504-full.uff
```
