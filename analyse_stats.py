import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
use_torque = False
use_noise = False

nb_try = 100
matcontent = sio.loadmat("solutions/stats_ac_activation_drivenTrue.mat")
err_ac = matcontent['err_tries']
err_ac_mhe = err_ac[:-1, :].reshape(-1, nb_try, 7)
err_ac_full = err_ac[-1, :].reshape(1, 7)
err_mean_ac = np.mean(err_ac_mhe, axis=1)
err_std_ac = np.std(err_ac_mhe, axis=1)
err_mean_ac_full = np.concatenate((err_mean_ac, err_ac_full))
Nmhe_ac = (err_mean_ac_full[:, 0])
time_ac = (err_mean_ac_full[:, 1])
time_std_ac = err_std_ac[:, 1]
time_mean_ac = (err_mean_ac[:, 1])
err_q_ac = np.log10(err_mean_ac_full[:, 2])
err_std_q_ac = (err_std_ac[:, 2])
err_dq_ac = np.log10(err_mean_ac_full[:, 3])
err_std_dq_ac = np.log10(err_std_ac[:, 3])
# err_tau_ac = np.log10(err_mean_ac_full[:, 4])
err_muscles_ac = np.log10(err_mean_ac_full[:, 5])
err_std_muscles_ac = np.log10(err_std_ac[:, 5])
err_markers_ac = np.log10(err_mean_ac_full[:, 6])
err_std_markers_ac = np.log10(err_std_ac[:, 6])

matcontent = sio.loadmat("solutions/stats_ac_activation_drivenFalse.mat")
err_ex = matcontent['err_tries']
err_ex_mhe = err_ex[:-1, :].reshape(-1, nb_try, 7)
err_ex_full = err_ex[-1, :].reshape(1, 7)
err_mean_ex = np.mean(err_ex_mhe, axis=1)
err_std_ex = np.std(err_ac_mhe, axis=1)
err_mean_ex_full = np.concatenate((err_mean_ex, err_ex_full))
Nmhe_ex = (err_mean_ex_full[:, 0])
time_ex = (err_mean_ex_full[:, 1])
time_mean_ex = (err_mean_ex[:, 1])
time_std_ex = (err_std_ex[:, 1])
err_q_ex = np.log10(err_mean_ex_full[:, 2])
err_std_q_ex = np.log10(err_std_ex[:, 2])
err_dq_ex = np.log10(err_mean_ex_full[:, 3])
err_std_dq_ex = np.log10(err_std_ex[:, 3])
# err_tau_ex = np.log10(err_mean_ex_full[:, 4])
err_muscles_ex = np.log10(err_mean_ex_full[:, 5])
err_std_muscles_ex = np.log10(err_std_ex[:, 5])
err_markers_ex = np.log10(err_mean_ex_full[:, 6])
err_std_markers_ex = np.log10(err_std_ex[:, 6])

fig = plt.subplot()
plt.plot(1/time_mean_ac, label='activation driven', c='red')
plt.plot(1/time_mean_ex, label='excitation driven', c='blue')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(1/0.075, (len(Nmhe_ac)-1, 1)), '--', label='biofeedback standard', c='orange')
plt.fill_between(range(len(time_mean_ac)), 1/(time_mean_ac+time_std_ac), 1/(time_mean_ac-time_std_ac), facecolor='red', alpha=0.2)
plt.fill_between(range(len(time_mean_ex)), 1/(time_mean_ex+time_std_ex), 1/(time_mean_ex-time_std_ex), facecolor='blue', alpha=0.2)
fig.set_xticks(range(len(Nmhe_ac)-1))
fig.set_xticklabels(Nmhe_ac[:-1])
plt.legend()
plt.ylabel('Freq. (Hz)')
plt.xlabel('Size of MHE window')

# Configure plot
err_ac = 'r-x'
err_ex = 'b-^'
err_full_ac = 'r--'
err_full_ex = 'b:'
lw = 0.8
ms_ac = 4
ms_ex = 2
mew = 0.01
err_lim = 'k-.'

fig = plt.figure()
fig = plt.subplot(511)
plt.plot(range(24), err_q_ac[:-1], err_ac, lw=lw, ms=ms_ac, label='err. mhe activation driven')
# plt.errorbar(range(24), err_q_ac[:-1], yerr=err_std_q_ac*2)
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_q_ac[-1], (len(Nmhe_ac)-1, 1)), err_full_ac, label='err. full window activation driven')
plt.plot(err_q_ex[:-1], err_ex, lw=lw, ms=ms_ac, mew=mew, label='err. mhe excitation driven')
# plt.errorbar(err_std_q_ex*2, 'b')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_q_ex[-1], (len(Nmhe_ac)-1, 1)), err_full_ex, label='err. full window excitation driven')
# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.array(1e-3), (len(Nmhe_ac)-1, 1)), c='red', label = 'limit_err_1e-3')
# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.array(1e-4), (len(Nmhe_ac)-1, 1)), c='purple', label = 'limit_err_1e-4')
# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.array(1e-5), (len(Nmhe_ac)-1, 1)), 'k', label = 'limit_err_1e-5')

fig.set_xticks(range(len(Nmhe_ac)-1))
fig.set_xticklabels(Nmhe_ac[:-1])
plt.ylabel('q err. (rad)')
# plt.xlabel('Size of MHE window')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

fig = plt.subplot(512)
plt.plot(err_dq_ac[:-1], err_ac, lw=lw, ms=ms_ac, label='err. mhe activation driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_dq_ac[-1], (len(Nmhe_ac)-1, 1)), err_full_ac,  label='err. full window activation driven')
plt.plot(err_dq_ex[:-1], err_ex, lw=lw, ms=ms_ac, mew=mew, label='err. mhe excitation driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_dq_ex[-1], (len(Nmhe_ac)-1, 1)), err_full_ex,  label='err. full window excitation driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.log10(np.array(1e-3)), (len(Nmhe_ac)-1, 1)), err_lim, lw=lw, label = 'limit_err_1e-3')
# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.array(1e-4), (len(Nmhe_ac)-1, 1)), c='purple', label = 'limit_err_1e-4')
# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.array(1e-5), (len(Nmhe_ac)-1, 1)), 'k', label = 'limit_err_1e-5')
fig.set_xticks(range(len(Nmhe_ac)-1))
fig.set_xticklabels(Nmhe_ac[:-1])
plt.ylabel('dq err. (rad/s)')
# plt.xlabel('Size of MHE window')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

# fig = plt.subplot(513)
# plt.plot(err_tau[:-1], 'x', label='err. mhe')
# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_tau[-1], (len(Nmhe_ac)-1, 1)), '--',  label='err. full window')
# fig.set_xticks(range(len(Nmhe_ac)-1))
# fig.set_xticklabels(Nmhe_ac[:-1])
# plt.ylabel('Tau err. (Nm)')
# plt.xlabel('Size of MHE window')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

fig = plt.subplot(513)
plt.plot(err_muscles_ac[:-1], err_ac, lw=lw, ms=ms_ac, label='err. mhe activation driven')
plt.plot(err_muscles_ex[:-1], err_ex, lw=lw, ms=ms_ac, mew=mew, label='err. mhe excitation driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_muscles_ac[-1], (len(Nmhe_ac)-1, 1)), err_full_ac,  label='err. full window activation driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_muscles_ex[-1], (len(Nmhe_ac)-1, 1)), err_full_ex,  label='err. full window excitation driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.log10(np.array(1e-3)), (len(Nmhe_ac)-1, 1)), err_lim, lw=lw, label = 'limit_err_1e-3')
# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.array(1e-4), (len(Nmhe_ac)-1, 1)), c='purple', label = 'limit_err_1e-4')
# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.array(1e-5), (len(Nmhe_ac)-1, 1)), 'k', label = 'limit_err_1e-5')
fig.set_xticks(range(len(Nmhe_ac)-1))
fig.set_xticklabels(Nmhe_ac[:-1])
plt.ylabel('Muscle act. err.')
# plt.xlabel('Size of MHE window')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

fig = plt.subplot(514)
plt.plot(err_markers_ac[:-1], err_ac, lw=lw, ms=ms_ac, label='err. mhe activation driven')
plt.plot(err_markers_ex[:-1], err_ex, lw=lw, ms=ms_ac, mew=mew, label='err. mhe excitation driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_markers_ac[-1], (len(Nmhe_ac)-1, 1)), err_full_ac  , lw=lw,  label='err. full window activation driven')
plt.plot(np.arange(len(Nmhe_ex)-1), np.tile(err_markers_ex[-1], (len(Nmhe_ac)-1, 1)), err_full_ex , lw=lw,  label='err. full window excitatiojn driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.log10(np.array(1e-3)), (len(Nmhe_ac)-1, 1)), err_lim, lw=lw, label = 'limit_err_1e-3')
# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.array(1e-4), (len(Nmhe_ac)-1, 1)), c='purple', label = 'limit_err_1e-4')
# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.array(1e-5), (len(Nmhe_ac)-1, 1)), 'k', label = 'limit_err_1e-5')
fig.set_xticks(range(len(Nmhe_ac)-1))
fig.set_xticklabels(Nmhe_ac[:-1])
plt.ylabel('Marker err. (m)')
plt.xlabel('Size of MHE window')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.show()