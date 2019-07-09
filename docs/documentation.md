# Kalman Filter
The Kalman Filter has two components to it.

The state vector which indicates the current state
and the covariance matrix.

Systems can have external influences that affect the result.

A non-linear system cannot use a kalman filter. We need an extended kalman filter.
An extended kalman filter relies on a linearized estimation of the original non-linear system.

The EKF needs to use the time propagation of the original equation. The kalman gain and covariance is calculated using the linearized system.
