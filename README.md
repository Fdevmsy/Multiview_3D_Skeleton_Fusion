## Multi-View 3D Joints Fusion

### Introduction

Python scripts for multi-view 3D skeletions fusion by heatmap-weighted average. 

### Requirements
- Python3
- matplotlib
- mpl_toolkits

### Usage
Input:

- Paths of two j3d npy file, each of them is in shape (N, 17, 3)
- Fill the path at line 233 and line 234

Output:

- A merged j3d saved in merged.npy in the same folder.
- The shape will be (N, 17, 3)

Run:
`Python mmview-merge.py`
