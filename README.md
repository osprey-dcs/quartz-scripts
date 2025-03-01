# Quartz data file reading


## Pre-requisites

- scipy
- h5py

## File formats

- `.hdr` - Acqusition header
  - `.dat` - Raw chassis sample data
- `.uff` - UFF58b

## Read full data sets

```py
from matplotlib.pyplot import *
import quartz
with quartz.open('some.uff') as F:
    sig = F['sigName']
    plot(sig.time, sig)
```

```py
from matplotlib.pyplot import *
import quartz
with quartz.open('some.hdr') as F:
    sig = F['sigName']
    plot(sig.time, sig)
```

## Read a raw samples

```py
from matplotlib.pyplot import *
from quartz import psc
with quartz.open('some.hdr') as F:
    sig = F['1'] # "name" '1' -> '32'
    plot(sig.time, sig)

    sig = F[0] # zero index
    plot(sig.time, sig)
