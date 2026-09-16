from pymlff import MLAB

ab = MLAB.from_file("data/raw/ML_AB")

ab.write_extxyz("data/ML_AB.xyz", "kbar")
