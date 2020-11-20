import pickle
import bioviz
import matplotlib.pyplot as plt
import biorbd
import scipy.io as sio
import numpy as np
import csv
import pandas as pd
import seaborn
from matplotlib.colors import LogNorm
import pingouin as pg
from utils import convert_txt_output_to_list

seaborn.set_style("whitegrid")
seaborn.color_palette()

biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
T = 8
Ns = 100
motion = "REACH2"
nb_try = 30
marker_noise_lvl = [0, 0.002, 0.005, 0.01]
EMG_noise_lvl = [0, 0.6, 1, 1.5, 0]
# EMG_noise_lvl = [0, 0]
EMG_lvl_label = ['track, n_lvl=0', 'track, n_lvl=0.6', 'track, n_lvl=1',
                 'track, n_lvl=1.5', 'track, n_lvl=2', 'track, n_lvl=2.5', 'minimize']
# EMG_lvl_label = ['track', 'minimize']
states_controls = ['q', 'dq', 'act', 'exc']
co_lvl = 4
co_lvl_label = ['None', 'low', 'mid', 'high']
RMSEmin = np.ndarray((co_lvl * len(marker_noise_lvl) * len(EMG_noise_lvl) * 4 * nb_try))
RMSEtrack = np.ndarray((co_lvl * len(marker_noise_lvl) * len(EMG_noise_lvl) * 4 * nb_try))

W_LOW_WEIGHTS = False
folder_w_track = "solutions/w_track_emg_rt"
folder_wt_track = "solutions/wt_track_emg_rt"
status_trackEMG = convert_txt_output_to_list(folder_w_track+'/status_track_rt_EMGTrue.txt',
                           co_lvl, len(marker_noise_lvl), len(EMG_noise_lvl), nb_try)
status_minEMG = convert_txt_output_to_list(folder_wt_track+'/status_track_rt_EMGFalse.txt',
                           co_lvl, len(marker_noise_lvl), len(EMG_noise_lvl), nb_try)

co_lvl_df = [co_lvl_label[0]]*len(marker_noise_lvl)*len(EMG_noise_lvl)*4*nb_try \
            + [co_lvl_label[1]]*len(marker_noise_lvl)*len(EMG_noise_lvl)*4*nb_try \
            + [co_lvl_label[2]]*len(marker_noise_lvl)*len(EMG_noise_lvl)*4*nb_try \
            + [co_lvl_label[3]]*len(marker_noise_lvl)*len(EMG_noise_lvl)*4*nb_try

marker_n_lvl_df = ([marker_noise_lvl[0]]*len(EMG_noise_lvl)*4*nb_try
                   + [marker_noise_lvl[1]]*len(EMG_noise_lvl)*4*nb_try
                   + [marker_noise_lvl[2]]*len(EMG_noise_lvl)*4*nb_try
                   + [marker_noise_lvl[3]]*len(EMG_noise_lvl)*4*nb_try)*co_lvl

EMG_n_lvl_df = ([EMG_lvl_label[0]]*4*nb_try + [EMG_lvl_label[1]]*4*nb_try
                + [EMG_lvl_label[2]]*4*nb_try + [EMG_lvl_label[3]]*4*nb_try
                + [EMG_lvl_label[4]]*4*nb_try)*co_lvl*len(marker_noise_lvl)

EMG_n_lvl_stats = (['track']*4*nb_try + ['track']*4*nb_try
                + ['track']*4*nb_try + ['track']*4*nb_try
                + ['minimize']*4*nb_try)*co_lvl*len(marker_noise_lvl)

# EMG_n_lvl_df = ([EMG_lvl_label[0]]*4*nb_try + [EMG_lvl_label[1]]*4*nb_try)*co_lvl*len(marker_noise_lvl)
#
# EMG_n_lvl_stats = (['track']*4*nb_try + ['minimize']*4*nb_try)*co_lvl*len(marker_noise_lvl)

states_controls_df = ([states_controls[0]]*nb_try + [states_controls[1]]*nb_try + [states_controls[2]]*nb_try
                      + [states_controls[3]]*nb_try)*co_lvl*len(marker_noise_lvl)*len(EMG_noise_lvl)
count = 0
count_nc_min = np.zeros((4, 4, 4))
count_nc_track = np.zeros((4, 4, 4))
for co in range(co_lvl):
    for marker_lvl in range(len(marker_noise_lvl)):
        for EMG_lvl in range(len(EMG_noise_lvl)):

            if EMG_lvl_label[EMG_lvl] == "minimize":
                mat_content = sio.loadmat(
                    f"{folder_wt_track}/track_mhe_wt_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                )
            else:
                mat_content = sio.loadmat(
                    f"{folder_w_track}/track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                )

            Nmhe = int(mat_content['N_mhe'])
            N = mat_content['N_tot']
            NS = int(N - Nmhe)

            X_est = mat_content['X_est']
            U_est = mat_content['U_est']
            q_ref = mat_content['x_sol'][:biorbd_model.nbQ(), ::3][:, :-Nmhe]
            dq_ref = mat_content['x_sol'][biorbd_model.nbQ():biorbd_model.nbQ() * 2, ::3][:, :-Nmhe]
            a_ref = mat_content['x_sol'][-biorbd_model.nbMuscles():, ::3][:, :-Nmhe]
            u_ref = mat_content['u_sol'][:, ::3][:, :-Nmhe]

            q_ref_try = np.ndarray((nb_try, q_ref.shape[0], q_ref.shape[1]))
            dq_ref_try = np.ndarray((nb_try, dq_ref.shape[0], dq_ref.shape[1]))
            a_ref_try = np.ndarray((nb_try, a_ref.shape[0], a_ref.shape[1]))
            u_ref_try = np.ndarray((nb_try, u_ref.shape[0], u_ref.shape[1]))

            for i in range(nb_try):
                if EMG_lvl_label[EMG_lvl] == "minimize":
                    if len(status_minEMG[co][marker_lvl][EMG_lvl][i]) > 10:
                        q_ref_try[i, :, :] = np.nan
                        dq_ref_try[i, :, :] = np.nan
                        a_ref_try[i, :, :] = np.nan
                        u_ref_try[i, :, :] = np.nan
                        count_nc_min[co, marker_lvl, EMG_lvl] += 1
                    else:
                        q_ref_try[i, :, :] = q_ref
                        dq_ref_try[i, :, :] = dq_ref
                        a_ref_try[i, :, :] = a_ref
                        u_ref_try[i, :, :] = u_ref
                else:
                    if len(status_trackEMG[co][marker_lvl][EMG_lvl][i]) > 20:
                        q_ref_try[i, :, :] = np.nan
                        dq_ref_try[i, :, :] = np.nan
                        a_ref_try[i, :, :] = np.nan
                        u_ref_try[i, :, :] = np.nan
                        count_nc_track[co, marker_lvl, EMG_lvl] += 1
                    else:
                        q_ref_try[i, :, :] = q_ref
                        dq_ref_try[i, :, :] = dq_ref
                        a_ref_try[i, :, :] = a_ref
                        u_ref_try[i, :, :] = u_ref

            Q_err = np.linalg.norm(X_est[:, :biorbd_model.nbQ(), :] - q_ref_try, axis=2) / np.sqrt(NS + 1)
            Q_err = np.nanmean(Q_err, axis=1)
            DQ_err = np.linalg.norm(
                X_est[:, biorbd_model.nbQ():biorbd_model.nbQ() * 2, :] - dq_ref_try, axis=2) / np.sqrt(NS + 1)
            DQ_err = np.nanmean(DQ_err, axis=1)
            A_err = np.linalg.norm(
                X_est[:, -biorbd_model.nbMuscles():, :] - a_ref_try, axis=2) / np.sqrt(NS + 1)
            A_err = np.nanmean(A_err, axis=1)
            U_err = np.linalg.norm(
                U_est[:, -biorbd_model.nbMuscles():, :] - u_ref_try, axis=2) / np.sqrt(NS)
            U_err = np.nanmean(U_err, axis=1)

            RMSEtrack[count:count+nb_try] = Q_err
            RMSEtrack[count+nb_try:count+2*nb_try] = DQ_err
            RMSEtrack[count+2*nb_try:count+3*nb_try] = A_err
            RMSEtrack[count+3*nb_try:count+4*nb_try] = U_err
            count += 4*nb_try

print(f"Number of optim: {int(count/5)}")
print(f"Number of optim convergence with EMG tracking: {count_nc_track.sum()}")
print(f"Number of optim convergence without EMG tracking: {count_nc_min.sum()}")

print(f"Convergence rate with EMG tracking: {100-count_nc_track/(count/5)*100}%")
print(f"Convergence rate without EMG tracking: {100-count_nc_min/(count/5)*100}%")

RMSEtrack_pd = pd.DataFrame({"RMSE": RMSEtrack, "co_contraction_level": co_lvl_df, "EMG_objective": EMG_n_lvl_df,
                             "Marker_noise_level_m": marker_n_lvl_df, "component": states_controls_df})

# STATS
df_stats = pd.DataFrame({"RMSE": RMSEtrack, "co_contraction_level": co_lvl_df, "EMG_objective": EMG_n_lvl_stats,
                             "Marker_noise_level_m": marker_n_lvl_df, "component": states_controls_df})
df_stats = df_stats[RMSEtrack_pd['component'] == 'exc']
df_stats = df_stats[df_stats['RMSE'].notna()]
df_stats.to_pickle('stats_df_1.pkl')

aov = pg.anova(dv='RMSE', between=['EMG_objective', 'co_contraction_level'],
               data=df_stats)
ptt = pg.pairwise_ttests(dv='RMSE', between=['EMG_objective', 'co_contraction_level'], data=df_stats, padjust='bonf')
pg.print_table(aov.round(3))
pg.print_table(ptt.round(3))


# PLOT

ax = seaborn.boxplot(y=RMSEtrack_pd['RMSE'][RMSEtrack_pd['component'] == 'exc'],
                     x=RMSEtrack_pd['co_contraction_level'],
                     hue=RMSEtrack_pd['EMG_objective'])

if W_LOW_WEIGHTS:
    title_str = "with lower weights on markers"
else:
    title_str = "with higher weights on markers"
ax.set(ylabel='RMSE on muscle excitations')
ax.xaxis.get_label().set_fontsize(20)
ax.yaxis.get_label().set_fontsize(20)
ax.legend(title='EMG objective')
ax.tick_params(labelsize=15)
plt.setp(ax.get_legend().get_texts(), fontsize='18') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='20') # for legend title
plt.title(f'Error on excitations {title_str}', fontsize=20)
plt.figure()

# STATS
df_stats = pd.DataFrame({"RMSE": RMSEtrack, "co_contraction_level": co_lvl_df, "EMG_objective": EMG_n_lvl_stats,
                             "Marker_noise_level_m": marker_n_lvl_df, "component": states_controls_df})
df_stats = df_stats[(RMSEtrack_pd['component'] == 'q')]
df_stats = df_stats[df_stats['RMSE'].notna()]
df_stats.to_pickle('stats_df_2.pkl')

aov = pg.anova(dv='RMSE', between=['Marker_noise_level_m', "EMG_objective"],
               data=df_stats, detailed=True)
ptt = pg.pairwise_ttests(dv='RMSE', between=['Marker_noise_level_m', "EMG_objective"], data=df_stats, padjust='bonf')
pg.print_table(aov.round(3))
pg.print_table(ptt.round(3))

# PLOT
#& RMSEtrack_pd['co-contraction level'] == 0
ax2 = seaborn.boxplot(y = RMSEtrack_pd['RMSE'][(RMSEtrack_pd['component'] == 'q')],
                      x = RMSEtrack_pd['Marker_noise_level_m'],
                      hue=RMSEtrack_pd['EMG_objective'],)
ax2.set(ylabel='RMSE on joint positions (rad)')
ax2.xaxis.get_label().set_fontsize(20)
ax2.yaxis.get_label().set_fontsize(20)
ax2.tick_params(labelsize=15)
ax2.legend(title='EMG objective')
plt.setp(ax2.get_legend().get_texts(), fontsize='18') # for legend text
plt.setp(ax2.get_legend().get_title(), fontsize='20') # for legend title
plt.title(f'Error on joint positions {title_str}', fontsize=20)
plt.show()