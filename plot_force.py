import biorbd
import scipy.io as sio
import numpy as np
import seaborn
from casadi import MX, Function
from math import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

seaborn.set_style("whitegrid")
seaborn.color_palette()
T = 8
start_delay = 25
Ns = 800 - start_delay
T = T * (Ns) / 800
motion = "REACH2"
nb_try = 30
marker_noise_lvl = [0, 0.002, 0.005, 0.01]
EMG_noise_lvl = [0, 1, 1.5, 2]
co_lvl = 4
fold_w_emg = f"solutions/w_track_emg_rt_exc/"
fold_wt_emg = f"solutions/wt_track_emg_rt_exc/"
status_trackEMG = convert_txt_output_to_list(fold_w_emg+'/status_track_rt_EMGTrue.txt',
                                             co_lvl, len(marker_noise_lvl), len(EMG_noise_lvl), nb_try)
count = 0
count_nc_track = 0
muscles_names = ['Pec sternal', 'Pec ribs', 'Lat thoracic', 'Lat lumbar', 'Lat iliac', 'Delt posterior',
                 'Tri long', 'Tri lat', 'Tri med', 'Brachial', 'Brachioradial', 'Pec clavicular', 'Delt anterior',
                 'Delt middle', 'Supraspin', 'Infraspin', 'Subscap', 'Bic long', 'Bic short']
biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
# for co in range(co_lvl):

count_nc_track = np.zeros((1,1,1))
for co in [2]:
    for marker_lvl in [2]:
        range_emg = [2]
        for EMG_lvl in range_emg:
            mat_content = sio.loadmat(
                f"{fold_w_emg}track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
            )
            Nmhe = int(mat_content['N_mhe'])
            N = mat_content['N_tot']
            NS = int(N - Nmhe)

            ratio = int(mat_content['rt_ratio'])
            X_est = mat_content['X_est']
            U_est = mat_content['U_est']
            f_est = mat_content['f_est']
            q_ref = mat_content['x_sol'][:biorbd_model.nbQ(), ::ratio][:, :-Nmhe]
            dq_ref = mat_content['x_sol'][biorbd_model.nbQ():biorbd_model.nbQ() * 2, ::ratio][:, :-Nmhe]
            a_ref = mat_content['x_sol'][-biorbd_model.nbMuscles():, ::ratio][:, :-Nmhe]
            u_ref = mat_content['u_sol'][:, ::ratio][:, :-Nmhe]
            f_ref = mat_content['f_ref'][:, ::ratio][:, :-Nmhe]

            q_ref_try = np.ndarray((nb_try, q_ref.shape[0], q_ref.shape[1]))
            dq_ref_try = np.ndarray((nb_try, dq_ref.shape[0], dq_ref.shape[1]))
            a_ref_try = np.ndarray((nb_try, a_ref.shape[0], a_ref.shape[1]))
            u_ref_try = np.ndarray((nb_try, u_ref.shape[0], u_ref.shape[1]))
            f_ref_try = np.ndarray((nb_try, f_ref.shape[0], f_ref.shape[1]))

            for i in range(nb_try):
                if len(status_trackEMG[co][marker_lvl][EMG_lvl][i]) > (5 * 252 / 100):
                    q_ref_try[i, :, :] = np.nan
                    dq_ref_try[i, :, :] = np.nan
                    a_ref_try[i, :, :] = np.nan
                    u_ref_try[i, :, :] = np.nan
                    f_ref_try[i, :, :] = np.nan
                    count_nc_track[0,0,0] += 1
                else:
                    q_ref_try[i, :, :] = q_ref
                    dq_ref_try[i, :, :] = dq_ref
                    a_ref_try[i, :, :] = a_ref
                    u_ref_try[i, :, :] = u_ref
                    f_ref_try[i, :, :] = f_ref
            ratio = int(mat_content['rt_ratio'])
            Nmhe = int(mat_content['N_mhe'])
            N = mat_content['N_tot']
            NS = int(N - Nmhe)
            force_est = mat_content['f_est']
            force_ref = mat_content['f_ref'][:, ::ratio][:, :-Nmhe]
            force_est_mean = np.mean(force_est, axis=0)
            force_est_STD = np.std(force_est, axis=0)

            mat_content = sio.loadmat(
                f"{fold_wt_emg}track_mhe_wt_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_0.mat"
            )
            ratio = int(mat_content['rt_ratio'])
            Nmhe = mat_content['N_mhe']
            N = mat_content['N_tot']
            NS = int(N - Nmhe)
            force_est_wt_emg = mat_content['f_est']
            force_est_wt_emg_mean = np.mean(force_est_wt_emg, axis=0)
            force_est_wt_emg_STD = np.std(force_est_wt_emg, axis=0)

            print(f"Number of optimisation: {ceil((Ns)/ratio-Nmhe)}")
            print(f"Number of optim convergence with EMG tracking: {count_nc_track}")
            print(f"Convergence rate with EMG tracking: {100 - count_nc_track / nb_try * 100}%")

            fig = plt.figure("Muscles force")
            plt.gcf().subplots_adjust(left=0.06, right=0.99, wspace=0.25, hspace=0.2)
            t = np.linspace(0, T, ceil((Ns)/ratio-Nmhe)-1)
            for i in range(biorbd_model.nbMuscles()):
                fig = plt.subplot(4, 5, i + 1)
                if i in [14, 15, 16, 17, 18]:
                    plt.xlabel('Time (s)')
                else:
                    fig.set_xticklabels([])
                if i in [0, 5, 10, 15]:
                    plt.ylabel('Muscle force (N)')

                # if i == 18:
                plt.plot(t, force_est_mean[i, 1:], label='Track EMG')
                plt.plot(t, force_est_wt_emg_mean[i, 1:],  label='Minimize EMG')
                plt.plot(t, force_ref[i, 1:], 'r', label='Reference')
                plt.gca().set_prop_cycle(None)
                plt.fill_between(t, force_est_mean[i, 1:] - force_est_STD[i, 1:],
                                 force_est_mean[i, 1:] + force_est_STD[i, 1:],
                                 alpha=0.5)
                plt.title(muscles_names[i])
            plt.legend(bbox_to_anchor=(1.05, 0.80), loc='upper left', frameon=False)
            plt.show()

