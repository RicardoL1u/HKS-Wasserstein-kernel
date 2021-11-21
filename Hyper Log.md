# Hyperparam logging

# HKS scaling Random
1205 10 Mean 10-fold accuracy: 75.03 +- 11.24 %
1205 100 Mean 10-fold accuracy: 82.40 +- 7.26 %
1205 400 Mean 10-fold accuracy: 82.95 +- 7.52 %
1205 1000 Mean 10-fold accuracy: 83.98 +- 8.47 %
1205 1250 Mean 10-fold accuracy: 84.50 +- 8.63 %
1205 2500 Mean 10-fold accuracy: 83.39 +- 8.34 %
1205 5000 Mean 10-fold accuracy: 82.92 +- 7.18 %
1205 10000 Mean 10-fold accuracy: 82.40 +- 7.99 %

# HKS scaling uniform

1000 | 82.95 | 9.76
1250 | 82.95 | 9.76
3000 | 83.48 | 9.96
5000 | 83.48 | 9.96

# HKS with exponential dis
```python
def get_random_samples_based_exp(T=8,lambda_ = 1):
    np.random.seed(42)
    beta = 1/lambda_
    return np.random.exponential(scale=lambda_,size=(T))
```
弄反了
1205 | 1 | 85.09 | 6.61
1205 | dynamic | 85.64 | 8.50
542  | dynamic | 85.64 | 7.46
42   | dynamic | 84.59 | 7.21

# HKS with exponential dual dis
```python
def get_random_samples_based_exp_dual(T=8,lambda_ = 1):
    np.random.seed(42)
    beta = 1/lambda_
    samples_left = np.random.exponential(scale=beta,size=(int(T/2)))
    samples_right = 1250-np.random.exponential(scale=beta,size=(int(T/2)))
    # samples_right = np.maximum(45-np.random.exponential(scale=beta,size=(int(T/2))),zero_list)
    # nothing change
    return np.concatenate((samples_left,samples_right))
```
42 | dynamic | 1250 | 85.67 | 8.46
42 | dynamic | 50   | 88.27 | 6.20 !!!
42 | dynamic | 45   | 88.27 | 6.20 !!! - (h: 3-8)
42 | dynamic | 60   | 86.17 | 5.87 !
42 | dynamic | 40   | 87.22 | 8.25 !
542 | dynamic | 50   | 85.64 | 7.08 
542 | dynamic | 50   | 86.70 | 5.91 - (h:5-10) 
1205 | dynamic | 50   | 87.22 | 6.77 - (h:5-10) 
1205 | dynamic | 50   | 86.70 | 5.91 

# Single Hyper
h = 400
g = 0.001
c = 1000
87.22 | 7.54