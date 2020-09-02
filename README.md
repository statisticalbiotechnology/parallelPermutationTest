<img src="/images/parallelPermTest.png">

# Installation
```
pip install parallelPermutationTest
```
or
```
git clone https://github.com/statisticalbiotechnology/parallelPermutationTest.git
cd parallelPermutationTest
make
```

# Requirement

Numpy and CUDA

# Tutorial

## Integer values


```
import parallelPermutationTest as ppt
import numpy as np
import pandas as pd

n_samples = 1
n = m =  500

data = lambda n,n_samples : np.asarray([np.random.randint(0,n,n,dtype=np.int32) for _ in range(n_samples)])

np.random.seed(42)
A,B = data(n,1), data(n,1)
```

```
%time p_green_gpu = ppt.GreenIntCuda(A,B)
```

```
CPU times: user 4.74 s, sys: 56.1 ms, total: 4.79 s
Wall time: 4.78 s
````

## Small Dataset: Floating values

```
#Daily S&P500 data from 1986==>
url = "https://raw.githubusercontent.com/Patrick-David/Stocks_Significance_PHacking/master/spx.csv"
df = pd.read_csv(url,index_col='date', parse_dates=True)


daily_ret = df['close'].pct_change()
daily_ret.dropna(inplace=True)

mnthly_annu = daily_ret.resample('M').std()* np.sqrt(12)

dec_vol = mnthly_annu[mnthly_annu.index.month==12]
rest_vol = mnthly_annu[mnthly_annu.index.month!=12]
```

```
dec_vol.head(2)
```

```
date
1986-12-31    0.026474
1987-12-31    0.061435
Name: close, dtype: float64
```

```
(dec_vol.values.shape, rest_vol.values.shape)
```

```
(32,), (358,))
```

```
%time p = ppt.GreenFloatCuda(dec_vol.values, rest_vol.values, 500)
```
```
CPU times: user 18.1 ms, sys: 20 µs, total: 18.1 ms
Wall time: 17.3 ms
```

## Large Dataset: Floating values
```
NotTNP_df = pd.read_csv("experiment_data/experiment6/notTNPdf")
TNP_df = pd.read_csv("experiment_data/experiment6/TNPdf")
```

```
(TNP_df.shape, NotTNP_df.shape)
```
```
((8051, 26), (8051, 80))
```
```
n_bins = 100
ppt.GreenFloatCuda_memcheck(TNP_df.values, NotTNP_df.values, n_bins)
```
```
Warning: The data requires 23503.55Mib, and the GPU has 7718Mib available, so there is 15785.55Mib too little memory. Consider dividing the data into batches.
```
```
batch_size = int(TNP_df.shape[0] / 4)
%time p_values = ppt.GreenFloatCuda(TNP_df.values, NotTNP_df.values, 100, batch_size=batch_size)
```
```
CPU times: user 10.5 s, sys: 1.26 s, total: 11.8 s
Wall time: 11.8 s
```

## Authors

* **Markus Ekvall, Lukas Käll and Micheal Höhle** 

## Acknowledgments

* Pagano and Tritchler(1983), and Zimmerman (1985) for unparalleled version of the shift-method.
* hnilsson: https://github.com/cran/coin

