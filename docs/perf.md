# Performance against NumPy 2.5.4

One core (GOMAXPROCS=1; OPENBLAS/OMP/MKL threads=1 for NumPy), Go go1.27.2 darwin/arm64, interleaved rounds (Go, NumPy, Go, NumPy, ...), median of 9 rounds. Target: no slower than NumPy (ratio <= 1).

Load before: `14:48  up 27 days,  1:47, 1 user, load averages: 6.73 7.80 8.48`

| operation | n (or degree) | Go | NumPy | Go / NumPy |
|---|---:|---:|---:|---:|
| polyval deg5 | 1000 | 1.4 µs | 9.1 µs | 0.16 |
| polyfit deg3 | 1000 | 13.2 µs | 70.4 µs | 0.19 |
| polyfit deg10 | 1000 | 47.8 µs | 148.6 µs | 0.32 |
| chebval 21 coef | 1000 | 8.7 µs | 37.0 µs | 0.24 |
| chebfit deg10 | 1000 | 58.0 µs | 146.0 µs | 0.40 |
| chebvander deg10 | 1000 | 7.4 µs | 21.2 µs | 0.35 |
| polyval deg5 | 10000 | 12.6 µs | 40.6 µs | 0.31 |
| polyfit deg3 | 10000 | 136.9 µs | 507.5 µs | 0.27 |
| polyfit deg10 | 10000 | 544.0 µs | 1.19 ms | 0.46 |
| chebval 21 coef | 10000 | 87.2 µs | 193.7 µs | 0.45 |
| chebfit deg10 | 10000 | 625.1 µs | 940.0 µs | 0.66 |
| chebvander deg10 | 10000 | 68.8 µs | 97.3 µs | 0.71 |
| polyval deg5 | 100000 | 117.4 µs | 287.5 µs | 0.41 |
| polyfit deg3 | 100000 | 1.34 ms | 4.50 ms | 0.30 |
| polyfit deg10 | 100000 | 5.22 ms | 10.21 ms | 0.51 |
| chebval 21 coef | 100000 | 859.7 µs | 1.43 ms | 0.60 |
| chebfit deg10 | 100000 | 6.12 ms | 7.25 ms | 0.84 |
| chebvander deg10 | 100000 | 687.9 µs | 755.0 µs | 0.91 |
| polyval deg5 | 1000000 | 1.29 ms | 2.86 ms | 0.45 |
| polyfit deg3 | 1000000 | 13.20 ms | 42.20 ms | 0.31 |
| polyfit deg10 | 1000000 | 53.60 ms | 92.60 ms | 0.58 |
| chebval 21 coef | 1000000 | 8.78 ms | 15.00 ms | 0.59 |
| chebfit deg10 | 1000000 | 60.08 ms | 63.14 ms | 0.95 |
| chebvander deg10 | 1000000 | 6.77 ms | 7.51 ms | 0.90 |
| roots | 10 | 13.7 µs | 25.0 µs | 0.55 |
| roots | 100 | 1.40 ms | 2.13 ms | 0.66 |

Load after: `14:50  up 27 days,  1:48, 1 user, load averages: 6.05 7.11 8.13`
