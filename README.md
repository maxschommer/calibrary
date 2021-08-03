# calibrary

A library of robot arm calibration and optimization procecdures

# DH Parameter Optimization

One of the primary optimizations this library provides is DH parameter optimization. All that is required to optimize DH parameters is a set of initial DH parameters, a set of known end effector poses (observed using a camera, laser tracker, etc.), and the associated joint angles.

```
from calibrary.calibrate_dh import optimize_dh

# An #Nx4x4 array of transformation matrices representing the end effector
# poses at each sample point.
poses = ...
# The initial DH parameters
dh_params = ...

# An #Nx6 array of joint angles associated with the poses.
joint_angles = ...
res_dh, err = optimize_dh(joint_angles, dh_params, poses)
```

The resulting DH parameters are returned, as well as the final error function
value.
