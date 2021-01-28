import matplotlib.pyplot as plt
import biorbd
import scipy.io as sio
import numpy as np
import pandas as pd
import seaborn
import pingouin as pg
from utils import convert_txt_output_to_list
from math import ceil

# Configure the plot
W_LOW_WEIGHTS = True

# Variables of the problem
biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
T = 8
Ns = 100
motion = "REACH2"
nb_try = 30
states_controls = ["q", "dq", "act", "exc", "force"]
co_lvl = 4


# Set folder of data

folder_w_track_lw = "solutions/w_track_emg_rt_exc_low_weight"
folder_wt_track_lw = "solutions/wt_track_emg_rt_exc_low_weight"
folder_w_track = "solutions/w_track_emg_rt_exc"
folder_wt_track = "solutions/wt_track_emg_rt_exc"

# Set noise parameters(same than used for the OCP) and labels
marker_noise_lvl = [0, 0.002, 0.005, 0.01]
EMG_noise_lvl = [0, 1, 1.5, 2, 0]
EMG_lvl_label = ["track/none", "track/low", "track/mid", "track/high", "minimize/-"]
co_lvl_label = ["none", "low", "mid", "high"]
marker_lvl_label = ["none", "low", "mid", "high"]
weight_lvl_label = ["low", "high"]

# Set RMSE size for minimize controls and track
RMSEmin = np.ndarray((len(weight_lvl_label) * co_lvl * len(marker_noise_lvl) * len(EMG_noise_lvl) * 5 * nb_try))
RMSEtrack = np.ndarray((len(weight_lvl_label) * co_lvl * len(marker_noise_lvl) * len(EMG_noise_lvl) * 5 * nb_try))

# Convert status files into a list to use it easily
status_trackEMG = convert_txt_output_to_list(
    folder_w_track + "/status_track_rt_EMGTrue.txt", co_lvl, len(marker_noise_lvl), len(EMG_noise_lvl), nb_try
)
status_minEMG = convert_txt_output_to_list(
    folder_wt_track + "/status_track_rt_EMGFalse.txt", co_lvl, len(marker_noise_lvl), len(EMG_noise_lvl), nb_try
)
status_trackEMG_lw = convert_txt_output_to_list(
    folder_w_track_lw + "/status_track_rt_EMGTrue.txt", co_lvl, len(marker_noise_lvl), len(EMG_noise_lvl), nb_try
)
status_minEMG_lw = convert_txt_output_to_list(
    folder_wt_track_lw + "/status_track_rt_EMGFalse.txt", co_lvl, len(marker_noise_lvl), len(EMG_noise_lvl), nb_try
)

# DataFrame stuff
weight_lvl_df = [weight_lvl_label[0]] * co_lvl * len(marker_noise_lvl) * len(EMG_noise_lvl) * 5 * nb_try + [
    weight_lvl_label[1]
] * co_lvl * len(marker_noise_lvl) * len(EMG_noise_lvl) * 5 * nb_try

co_lvl_df = (
    [co_lvl_label[0]] * len(marker_noise_lvl) * len(EMG_noise_lvl) * 5 * nb_try
    + [co_lvl_label[1]] * len(marker_noise_lvl) * len(EMG_noise_lvl) * 5 * nb_try
    + [co_lvl_label[2]] * len(marker_noise_lvl) * len(EMG_noise_lvl) * 5 * nb_try
    + [co_lvl_label[3]] * len(marker_noise_lvl) * len(EMG_noise_lvl) * 5 * nb_try
) * len(weight_lvl_label)

marker_n_lvl_df = (
    (
        [marker_lvl_label[0]] * len(EMG_noise_lvl) * 5 * nb_try
        + [marker_lvl_label[1]] * len(EMG_noise_lvl) * 5 * nb_try
        + [marker_lvl_label[2]] * len(EMG_noise_lvl) * 5 * nb_try
        + [marker_lvl_label[3]] * len(EMG_noise_lvl) * 5 * nb_try
    )
    * len(weight_lvl_label)
    * co_lvl
)

EMG_n_lvl_df = (
    (
        [EMG_lvl_label[0]] * 5 * nb_try
        + [EMG_lvl_label[1]] * 5 * nb_try
        + [EMG_lvl_label[2]] * 5 * nb_try
        + [EMG_lvl_label[3]] * 5 * nb_try
        + [EMG_lvl_label[4]] * 5 * nb_try
    )
    * len(weight_lvl_label)
    * co_lvl
    * len(marker_noise_lvl)
)

states_controls_df = (
    (
        [states_controls[0]] * nb_try
        + [states_controls[1]] * nb_try
        + [states_controls[2]] * nb_try
        + [states_controls[3]] * nb_try
        + [states_controls[4]] * nb_try
    )
    * len(weight_lvl_label)
    * co_lvl
    * len(marker_noise_lvl)
    * len(EMG_noise_lvl)
)

# Run for all co-contraction and noise level
count = 0
count_nc_min = np.zeros((co_lvl, len(marker_noise_lvl), len(EMG_noise_lvl)))
count_nc_track = np.zeros((co_lvl, len(marker_noise_lvl), len(EMG_noise_lvl)))
nb_optim_track = co_lvl * len(marker_noise_lvl) * len(EMG_noise_lvl) * nb_try
for weight in weight_lvl_label:
    for co in range(co_lvl):
        for marker_lvl in range(len(marker_noise_lvl)):
            for EMG_lvl in range(len(EMG_noise_lvl)):
                # Get data
                if EMG_lvl_label[EMG_lvl] == "minimize/-":
                    if weight == "low":
                        mat_content = sio.loadmat(
                            f"{folder_wt_track_lw}/track_mhe_wt_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                        )
                    else:
                        mat_content = sio.loadmat(
                            f"{folder_wt_track}/track_mhe_wt_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                        )
                else:
                    if weight == "low":
                        mat_content = sio.loadmat(
                            f"{folder_w_track_lw}/track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                        )
                    else:
                        mat_content = sio.loadmat(
                            f"{folder_w_track}/track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                        )

                # Store data in variables
                Nmhe = int(mat_content["N_mhe"])
                N = mat_content["N_tot"]
                NS = int(N - Nmhe)

                ratio = int(mat_content["rt_ratio"])
                X_est = mat_content["X_est"]
                U_est = mat_content["U_est"]
                f_est = mat_content["f_est"]
                q_ref = mat_content["x_sol"][: biorbd_model.nbQ(), ::ratio][:, :-Nmhe]
                dq_ref = mat_content["x_sol"][biorbd_model.nbQ() : biorbd_model.nbQ() * 2, ::ratio][:, :-Nmhe]
                a_ref = mat_content["x_sol"][-biorbd_model.nbMuscles() :, ::ratio][:, :-Nmhe]
                u_ref = mat_content["u_sol"][:, ::ratio][:, :-Nmhe]
                f_ref = mat_content["f_ref"][:, ::ratio][:, :-Nmhe]

                q_ref_try = np.ndarray((nb_try, q_ref.shape[0], q_ref.shape[1]))
                dq_ref_try = np.ndarray((nb_try, dq_ref.shape[0], dq_ref.shape[1]))
                a_ref_try = np.ndarray((nb_try, a_ref.shape[0], a_ref.shape[1]))
                u_ref_try = np.ndarray((nb_try, u_ref.shape[0], u_ref.shape[1]))
                f_ref_try = np.ndarray((nb_try, f_ref.shape[0], f_ref.shape[1]))

                # Trials with more than 90% of the optimisations which have converged will be kept, the others will be
                # filled with NaN values
                for i in range(nb_try):
                    if EMG_lvl_label[EMG_lvl] == "minimize":
                        if weight == "low":
                            if len(status_minEMG_lw[co][marker_lvl][EMG_lvl][i]) > (
                                10 * ceil((N) / ratio - Nmhe) / 100
                            ):
                                q_ref_try[i, :, :] = np.nan
                                dq_ref_try[i, :, :] = np.nan
                                a_ref_try[i, :, :] = np.nan
                                u_ref_try[i, :, :] = np.nan
                                f_ref_try[i, :, :] = np.nan
                                count_nc_min[co, marker_lvl, EMG_lvl] += 1
                            else:
                                q_ref_try[i, :, :] = q_ref
                                dq_ref_try[i, :, :] = dq_ref
                                a_ref_try[i, :, :] = a_ref
                                u_ref_try[i, :, :] = u_ref
                                f_ref_try[i, :, :] = f_ref
                        if weight == "high":
                            if len(status_minEMG[co][marker_lvl][EMG_lvl][i]) > (10 * ceil((N) / ratio - Nmhe) / 100):
                                q_ref_try[i, :, :] = np.nan
                                dq_ref_try[i, :, :] = np.nan
                                a_ref_try[i, :, :] = np.nan
                                u_ref_try[i, :, :] = np.nan
                                f_ref_try[i, :, :] = np.nan
                                count_nc_min[co, marker_lvl, EMG_lvl] += 1
                            else:
                                q_ref_try[i, :, :] = q_ref
                                dq_ref_try[i, :, :] = dq_ref
                                a_ref_try[i, :, :] = a_ref
                                u_ref_try[i, :, :] = u_ref
                    else:
                        if weight == "low":
                            if len(status_trackEMG_lw[co][marker_lvl][EMG_lvl][i]) > (
                                10 * ceil((N) / ratio - Nmhe) / 100
                            ):
                                q_ref_try[i, :, :] = np.nan
                                dq_ref_try[i, :, :] = np.nan
                                a_ref_try[i, :, :] = np.nan
                                u_ref_try[i, :, :] = np.nan
                                f_ref_try[i, :, :] = np.nan
                                count_nc_track[co, marker_lvl, EMG_lvl] += 1
                            else:
                                q_ref_try[i, :, :] = q_ref
                                dq_ref_try[i, :, :] = dq_ref
                                a_ref_try[i, :, :] = a_ref
                                u_ref_try[i, :, :] = u_ref
                                f_ref_try[i, :, :] = f_ref
                        else:
                            if len(status_trackEMG[co][marker_lvl][EMG_lvl][i]) > (10 * ceil((N) / ratio - Nmhe) / 100):
                                q_ref_try[i, :, :] = np.nan
                                dq_ref_try[i, :, :] = np.nan
                                a_ref_try[i, :, :] = np.nan
                                u_ref_try[i, :, :] = np.nan
                                f_ref_try[i, :, :] = np.nan
                                count_nc_track[co, marker_lvl, EMG_lvl] += 1
                            else:
                                q_ref_try[i, :, :] = q_ref
                                dq_ref_try[i, :, :] = dq_ref
                                a_ref_try[i, :, :] = a_ref
                                u_ref_try[i, :, :] = u_ref
                                f_ref_try[i, :, :] = f_ref

                # Computing RMSE
                Q_err = np.linalg.norm(X_est[:, : biorbd_model.nbQ(), :] - q_ref_try, axis=2) / np.sqrt(NS + 1)
                Q_err = np.nanmean(Q_err, axis=1)
                DQ_err = np.linalg.norm(
                    X_est[:, biorbd_model.nbQ() : biorbd_model.nbQ() * 2, :] - dq_ref_try, axis=2
                ) / np.sqrt(NS + 1)
                DQ_err = np.nanmean(DQ_err, axis=1)
                A_err = np.linalg.norm(X_est[:, -biorbd_model.nbMuscles() :, :] - a_ref_try, axis=2) / np.sqrt(NS + 1)
                A_err = np.nanmean(A_err, axis=1)
                U_err = np.linalg.norm(U_est[:, -biorbd_model.nbMuscles() :, :] - u_ref_try, axis=2) / np.sqrt(NS)
                U_err = np.nanmean(U_err, axis=1)
                F_err = np.linalg.norm(f_est[:, -biorbd_model.nbMuscles() :, :] - f_ref_try, axis=2) / np.sqrt(NS)
                F_err = np.nanmean(F_err, axis=1)

                RMSEtrack[count : count + nb_try] = Q_err * 180 / np.pi
                RMSEtrack[count + nb_try : count + 2 * nb_try] = DQ_err
                RMSEtrack[count + 2 * nb_try : count + 3 * nb_try] = A_err
                RMSEtrack[count + 3 * nb_try : count + 4 * nb_try] = U_err
                RMSEtrack[count + 4 * nb_try : count + 5 * nb_try] = F_err
                count += 5 * nb_try

# Print informations obout convergent statistics
print(f"Number of optim: {int(count/5)}")
print(f"Number of optim convergence with EMG tracking: {count_nc_track.sum()}")
print(f"Number of optim convergence without EMG tracking: {count_nc_min.sum()}")
print(f"Total convergence rate with EMG tracking : {100 - count_nc_track.sum() * 100 /nb_optim_track}")
print(f"Total convergence rate with EMG tracking : {100 - count_nc_min.sum() * 100 / (count / 5 - nb_optim_track)}")
print(f"Convergence rate with EMG tracking: {100-count_nc_track/nb_try*100}%")
print(f"Convergence rate without EMG tracking: {100-count_nc_min/nb_try*100}%")

# Store results in a DataFrame
RMSEtrack_pd = pd.DataFrame(
    {
        "RMSE": RMSEtrack,
        "co_contraction_level": co_lvl_df,
        "EMG_objective": EMG_n_lvl_df,
        "Marker_noise_level_m": marker_n_lvl_df,
        "component": states_controls_df,
        "weight_level": weight_lvl_df,
    }
)

# Statistical analysis on excitations with an Anova method
df_stats = pd.DataFrame(
    {
        "RMSE": RMSEtrack,
        "co_contraction_level": co_lvl_df,
        "EMG_objective": EMG_n_lvl_df,
        "Marker_noise_level_m": marker_n_lvl_df,
        "component": states_controls_df,
        "weight_level": weight_lvl_df,
    }
)
df_stats = df_stats[df_stats["component"] == "force"]
df_stats = df_stats[df_stats["weight_level"] == "high"]

df_stats = df_stats[df_stats["RMSE"].notna()]
df_stats.to_pickle("stats_df_1.pkl")
aov = pg.anova(dv="RMSE", between=["EMG_objective", "co_contraction_level"], data=df_stats)
ptt = pg.pairwise_ttests(
    dv="RMSE",
    between=[
        "co_contraction_level",
        "EMG_objective",
    ],
    data=df_stats,
    padjust="bonf",
)
pg.print_table(aov.round(3))
pg.print_table(ptt.round(3))

# Figure of RMSE on force function of co-contraction level (Fig. 7)
import matplotlib

matplotlib.rcParams["legend.handlelength"] = 4
matplotlib.rcParams["legend.handleheight"] = 2.25
seaborn.set_style("whitegrid")
cp = seaborn.color_palette("YlOrRd", 5)
cp[-1] = (0, 102 / 255, 153 / 255)
plotpd = RMSEtrack_pd[RMSEtrack_pd["component"] == "force"]
plotpd = plotpd[plotpd["weight_level"] == "high"]

ax = seaborn.boxplot(
    y=plotpd["RMSE"],
    x=plotpd["co_contraction_level"],
    hue=plotpd["EMG_objective"],
    palette=cp,
)

ax.set(ylabel="RMSE on muscle force (N)")
ax.set(xlabel="Co-contraction level")
ax.xaxis.get_label().set_fontsize(30)
ax.yaxis.get_label().set_fontsize(30)
ax.legend(title="Objective / EMG noise level")
ax.tick_params(labelsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize="22")  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize="30")  # for legend title
# plt.title(f"RMSE on muscle forces", fontsize=20)
plt.figure()

# Statistical analysis on states with an Anova method
df_stats = pd.DataFrame(
    {
        "RMSE": RMSEtrack,
        "co_contraction_level": co_lvl_df,
        "EMG_objective": EMG_n_lvl_df,
        "Marker_noise_level_m": marker_n_lvl_df,
        "component": states_controls_df,
        "weight_level": weight_lvl_df,
    }
)
df_stats = df_stats[(RMSEtrack_pd["component"] == "q")]
df_stats = df_stats[df_stats["RMSE"].notna()]
df_stats.to_pickle("stats_df_2.pkl")
aov = pg.anova(
    dv="RMSE", between=["Marker_noise_level_m", "EMG_objective", "weight_level"], data=df_stats, detailed=True
)
pg.print_table(aov.round(3))

df_stats_lw = df_stats[df_stats["weight_level"] == "low"]
df_stats_hw = df_stats[df_stats["weight_level"] == "high"]

ptt = pg.pairwise_ttests(dv="RMSE", between=["Marker_noise_level_m", "EMG_objective"], data=df_stats_hw, padjust="bonf")
pg.print_table(ptt.round(3))
ptt = pg.pairwise_ttests(dv="RMSE", between=["EMG_objective", "Marker_noise_level_m"], data=df_stats_hw, padjust="bonf")
pg.print_table(ptt.round(3))
ptt = pg.pairwise_ttests(dv="RMSE", between=["Marker_noise_level_m", "EMG_objective"], data=df_stats_lw, padjust="bonf")
pg.print_table(ptt.round(3))
ptt = pg.pairwise_ttests(dv="RMSE", between=["EMG_objective", "Marker_noise_level_m"], data=df_stats_lw, padjust="bonf")
pg.print_table(ptt.round(3))


# Figure of RMSE on joint angle function of marker noise level (Fig. 8-9)
plotpd = RMSEtrack_pd[RMSEtrack_pd["component"] == "q"]
plotpd_hw = plotpd[plotpd["weight_level"] == "high"]
plotpd_lw = plotpd[plotpd["weight_level"] == "low"]


ax2 = seaborn.boxplot(
    y=plotpd_hw["RMSE"],
    x=plotpd_hw["Marker_noise_level_m"],
    hue=plotpd_hw["EMG_objective"],
    palette=cp,
)
ax2.set(ylabel="RMSE on joint angle (deg)")
ax2.set(xlabel="Marker noise level")
ax2.xaxis.get_label().set_fontsize(30)
ax2.yaxis.get_label().set_fontsize(30)
ax2.tick_params(labelsize=20)
ax2.legend(title="Objective / EMG noise level")
plt.setp(ax2.get_legend().get_texts(), fontsize="22")  # for legend text
plt.setp(ax2.get_legend().get_title(), fontsize="30")  # for legend title
plt.title(f"RMSE on joint angles with higher weights on markers (1e9)", fontsize=20)
plt.figure()

ax3 = seaborn.boxplot(
    y=plotpd_lw["RMSE"],
    x=plotpd_lw["Marker_noise_level_m"],
    hue=plotpd_lw["EMG_objective"],
    palette=cp,
)
ax3.set(ylabel="RMSE on joint angle (deg)")
ax3.set(xlabel="Marker noise level")
ax3.xaxis.get_label().set_fontsize(30)
ax3.yaxis.get_label().set_fontsize(30)
ax3.tick_params(labelsize=20)
ax3.legend(title="Objective / EMG noise level")
plt.setp(ax3.get_legend().get_texts(), fontsize="22")  # for legend text
plt.setp(ax3.get_legend().get_title(), fontsize="30")  # for legend title
plt.title(f"RMSE on joint angles with lower weights on markers (1e7)", fontsize=20)
plt.ylim(0, 3)
plt.show()


plt.show()
