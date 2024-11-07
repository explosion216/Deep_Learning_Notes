# Chapter 3

## Step function

```python
def step_function(x):
    return np.array(x>0,dtype=np.int)
```

## Sigmoid

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

## ReLU

```python
def relu(x):
    return np.maximum(0,x)
```

## Softmax

```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

## index of the maximum element in each row

```python
np.argmax(x,axis=1)
```
