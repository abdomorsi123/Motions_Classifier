# Skeleton Data
The `.csv` files in the subdirectories all have the same structure.
Each line contains the keypoints and angles of exactly one pose/skeleton.
Each keypoint is represented by the absolute coordinates in a frame `x`, `y` and their confidence `c`.
In addition, each sample contains the elbow and armpit angle of the right and left side.
All videos were recorded with a resolution of `640x480`.

The order of the keypoints is the one [returned by OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-body_25)
using the *BOXY_25* model:

```json
{
    "0":    "Nose",
    "1":    "Neck",
    "2":    "RShoulder",
    "3":    "RElbow",
    "4":    "RWrist",
    "5":    "LShoulder",
    "6":    "LElbow",
    "7":    "LWrist",
    "8":    "MidHip",
    "9":    "RHip",
    "10":   "RKnee",
    "11":   "RAnkle",
    "12":   "LHip",
    "13":   "LKnee",
    "14":   "LAnkle",
    "15":   "REye",
    "16":   "LEye",
    "17":   "REar",
    "18":   "LEar",
    "19":   "LBigToe",
    "20":   "LSmallToe",
    "21":   "LHeel",
    "22":   "RBigToe",
    "23":   "RSmallToe",
    "24":   "RHeel",
}
```

So the indices to reference the keypoints and angles in a sample are

```
IDX_NOSE        = 0
IDX_NECK        = 3
IDX_R_SHOULDER  = 6
IDX_R_ELBOW     = 9
IDX_R_WRIST     = 12
IDX_L_SHOULDER  = 15
IDX_L_ELBOW     = 18
IDX_L_WRIST     = 21
IDX_M_HIP       = 24
IDX_R_HIP       = 27
IDX_R_KNEE      = 30
IDX_R_ANKLE     = 33
IDX_L_HIP       = 36
IDX_L_KNEE      = 39
IDX_L_ANKLE     = 42
IDX_R_EYE       = 45
IDX_L_EYE       = 48
IDX_R_EAR       = 51
IDX_L_EAR       = 54
IDX_L_BIG_TOE   = 57
IDX_L_SMALL_TOE = 60
IDX_L_HEEL      = 63
IDX_R_BIG_TOE   = 66
IDX_R_SMALL_TOE = 69
IDX_R_HEEL      = 72
IDX_R_ANGLE_ELBOW = 75
IDX_R_ANGLE_ARMPIT = 76
IDX_L_ANGLE_ELBOW = 77
IDX_L_ANGLE_ARMPIT = 78
```

## Folder `train`
This folder contains the skeleton data of the 27 participants constituting the training set. 
There are 5 possible actions: *Boxing*, *Drums*, *Guitar*, *Rowing* and *Violin*.
Each action has been performed approximately 10 times by each participant.
One file contains the data of one participant performing one action one time. 
For example, "p1_boxing_05.csv" contains the data of participant 1 performing the boxing gesture for the 5th time.

Training participants IDs: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,19,20,21,22,23,24,27,28,29,31

## Folder `test`
This folder contains the skeleton data of the 6 participants constituting the test set. 
None of the participants in this folder is part of the training set.
Data structure and file name structure are identical to the training set.

Test participants IDs: 18,25,26,30,32,33