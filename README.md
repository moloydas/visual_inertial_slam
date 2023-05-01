# Visual Inertial Slam

## Introduction
In this work, EKF-based Visual Inertial Simultaneous Localization And Mapping (SLAM) is implemented using IMU data as odometry and a stereo camera as our measurement sensor. It is a popular algorithm where the robot trajectory and the mapping of the landmarks is done simultaneously. The landmarks here are essentially point-like features that are scale-invariant and rotation-invariant with favourable geometric characteristics (for eg corner). These features could be coming from any detectors and must be post processed to remove outliers. The accuracy of this Algorithm will get impacted if outliers are present.

## Theoritical Assumptions:
- The prior probability is gaussian.
- The motion model is affected by gaussian noise.
- The observation model is affected by Gaussian noise.
- The process noise w t and measurement noise vt are independent of each other, of the state xt and across time.
- The posterior pdf is forced to be gaussian via approximation

## Assumptions in this implementation:
- Features and correspondances are provided
- all correspondances are true
- all the landmarks are their visibility is already provided for the entire trajectory

# Demo

##10 sequence

https://user-images.githubusercontent.com/20353960/235547234-31699c92-c3ef-49c6-8ad9-df5028ea4235.mp4

------

##03 sequence


https://user-images.githubusercontent.com/20353960/235547305-e9a99c28-2272-4d79-a80b-6af542087867.mp4


