# PyQt vs tkinter
Although both produce about the same amount of lines of code, Qt has a more
comfortable set of widgets to work with. It also comes with excellent
documentation. 

Qt is also much more modern than tkinter. The effort required to make tkinter
look good is much more than the effort required to make Qt look good.


# Building ECG
Creating an ecg occurs by building an ecg as described in this 
[paper](http://web.mit.edu/~gari/www/papers/ieeetbe50p289.pdf).


# Kalman Filter
The Kalman filter works under circumstances where you have an uncertain dynamic system
where you can make educated guesses about how the system will act.

To put it simpler, the kalman filter guesses the ending state of the system.

The variables inside a kalman filter must be gaussian distributed. 
This means that our variables have a `mu` (average, mean) and a `sigma2` (variance).

## State
The state is a list of variables for the current state of the system. 

A state for the system at k would be:
```
x_k = [position velocity]
```
where x_k contains the mean value for our gaussian distribution.


## Covariance matrix
We may have a correlation between the variables in our state. This information is captured in
what is called a covariance matrix.

The covariance matrix is in essence storing the values of our `sigma2` (variance) into a matrix.

A covariance matrix for our example above is:
```
pp pv
vp vv

where
    pp = pos pos, pv = pos vel
    vp = vel pos, vv = vel vel
```


## Prediction matrix
The kalman filter considers the k-1th state and predicts the a state for the kth.

A prediction matrix is required to move the k-1th state to the kth state.
Suppose we have a simple kinematic formula as follows:
```
p_k = p_k-1 + timechange v_k-1
v_k =         v_k-1
```

We set the values of our current state to one. So p_k-1 and v_k-1 = 1
```
1 timechange
0          1
```
This is our prediction matrix.

In addition to updating the state we also need to update our covariance matrix.
```
x_k = F_k * x_k-1
P_k = F_k * P_k-1 * F_k^T
where
    ^T indicates the identity of the matrix.
```


## External influence
External influence is added onto the state. An external influence consists of a control
matrix and a control vector.
```
x_k = F_k * x_k-1 + B_k * u_k
```


## External uncertainty
External influence is something that we already know about. External uncertainty is a force
that we don't know about.

This external uncertainty is added onto our covariance matrix.
```
P_k = F_k * P_k-1 * F_k^T + Q_k
```


## Refinement with Measurements
We can refine an estimate by providing measurements. These readings follow the same principle as the previous section. 
i.e. a state, covariance matrix and prediction matrix.

```
z_k = H_k * x_k
R_k = H_k * P_k * H_k^T
where
    H_k is the prediction matrix for the readings.
```

At this point we have two gaussian blobs, one from an estimate and another from measurements.

In order to refine our guess we should make a new guess which is combined from our estimate
and measurements.
What we end up doing is multiplying both gaussian blobs together to produce an overlap.

The rest is a bit fuzzy. We can find a kalman gain instantly or we can use some method that calculates the mean. Not sure what the correct math behind it is.


## References
Before continuing the contents here come from the following articles:
1. [How a Kalman Filter Works](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
2. [Kalman Filter Step By Step](https://towardsdatascience.com/kalman-filters-a-step-by-step-implementation-guide-in-python-91e7e123b968)
