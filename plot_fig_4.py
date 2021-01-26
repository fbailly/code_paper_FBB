import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import seaborn

# Configuration of problem
use_torque = False
use_noise = False

nb_try = 100  # Number of try used to generate data

# Get data
matcontent = sio.loadmat("solutions/stats_rt_activation_drivenTrue.mat")
err_ac = matcontent["err_tries"]
err_ac_mhe = err_ac[:-1, :].reshape(-1, nb_try, 10)
err_ac_full = err_ac[-1, :].reshape(1, 10)
err_mean_ac = np.mean(err_ac_mhe, axis=1)
err_std_ac = np.std(err_ac_mhe, axis=1)
err_mean_ac_full = np.concatenate((err_mean_ac, err_ac_full))
Nmhe_ac = err_mean_ac_full[:, 0]
ratio_ac = err_mean_ac_full[:-1, 1]
time_tot_ac = err_mean_ac_full[:, 2]
time_ac = err_mean_ac_full[:, 3]
time_std_ac = err_std_ac[:, 3]
time_mean_ac = err_mean_ac[:, 3]
nb_try = 100

matcontent = sio.loadmat("solutions/stats_rt_activation_drivenFalse.mat")
err_ex = matcontent["err_tries"]
err_ex_mhe = err_ex[:-1, :].reshape(-1, nb_try, 10)
err_ex_full = err_ex[-1, :].reshape(1, 10)
err_mean_ex = np.mean(err_ex_mhe, axis=1)
err_std_ex = np.std(err_ex_mhe, axis=1)
err_mean_ex_full = np.concatenate((err_mean_ex, err_ex_full))
Nmhe_ex = err_mean_ex_full[:, 0]
ratio_ex = err_mean_ex_full[:-1, 1]
time_tot_ex = err_mean_ex_full[:, 2]
time_ex = err_mean_ex_full[:, 3]
time_std_ex = err_std_ex[:, 3]
time_mean_ex = err_mean_ex[:, 3]

# Plot Frequency figure fonction of windows size
seaborn.set_style("whitegrid")
seaborn.color_palette()
lw = 8  # Line width
x_y_label = 30
x_y_ticks = 25
fig = plt.subplot()
fig.plot(1 / time_mean_ac, lw=lw, label="activation driven")
fig.plot(1 / time_mean_ex, lw=lw, label="excitation driven")
fig.plot(
    np.arange(len(Nmhe_ac) - 1), np.tile(1 / 0.075, (len(Nmhe_ac) - 1, 1)), "--", lw=lw, label="biofeedback standard"
)
fig.fill_between(
    range(len(time_mean_ac)), 1 / (time_mean_ac + time_std_ac), 1 / (time_mean_ac - time_std_ac), alpha=0.2
)
fig.fill_between(
    range(len(time_mean_ex)), 1 / (time_mean_ex + time_std_ex), 1 / (time_mean_ex - time_std_ex), alpha=0.2
)
fig.set_xticks(range(len(Nmhe_ac) - 1))
plt.xticks(fontsize=x_y_ticks)
plt.yticks(fontsize=x_y_ticks)
fig.set_xticklabels(range(int(Nmhe_ac[0]), int(Nmhe_ac[-2] + 1)))
plt.legend(fontsize=x_y_label)
plt.ylabel("Freq. (Hz)", fontsize=x_y_label)
plt.xlabel("Size of MHE window", fontsize=x_y_label)
plt.show()
