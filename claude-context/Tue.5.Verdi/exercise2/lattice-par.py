import sys
import numpy as np

name = sys.argv[1]
T = float(sys.argv[2])
t_start = float(sys.argv[3])   # discard data before t_start (ps)
log_file = f"{name}-T{T}.log"

data = np.loadtxt(log_file, comments="#")
time = data[:, 0]
a = data[:, 5]
b = data[:, 6]
c = data[:, 7]

mask = time >= t_start
a_ave = np.mean(a[mask]); a_std = np.std(a[mask])
b_ave = np.mean(b[mask]); b_std = np.std(b[mask])
c_ave = np.mean(c[mask]); c_std = np.std(c[mask])

print(f"File: {log_file}")
print(f"Averaging from time >= {t_start} ps")
print(f"a_avg = {a_ave:12.4f} Å   std = {a_std:12.4f} Å")
print(f"b_avg = {b_ave:12.4f} Å   std = {b_std:12.4f} Å")
print(f"c_avg = {c_ave:12.4f} Å   std = {c_std:12.4f} Å")
