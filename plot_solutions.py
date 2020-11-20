import pickle
import bioviz
import matplotlib.pyplot as plt
import biorbd
import scipy.io as sio
import numpy as np
import csv
import pandas as pd
import seaborn
import matplotlib
from matplotlib.colors import LogNorm

biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
T = 0.8
Ns = 100
motion = "REACH2"
nb_try = 1
marker_noise_lvl = [0, 0.002, 0.005, 0.01]
EMG_noise_lvl = [0, 0.05, 0.1, 0.2]
co_lvl = 4
RMSE = np.ndarray((co_lvl, len(marker_noise_lvl), 4 * len(EMG_noise_lvl)))
STD = np.ndarray((co_lvl, len(marker_noise_lvl), 4 * len(EMG_noise_lvl)))
count = 0
for co in range(co_lvl):
    for marker_lvl in range(len(marker_noise_lvl)):
        count = 0
        for EMG_lvl in range(len(EMG_noise_lvl)):
            mat_content = sio.loadmat(
                f"solutions/w_track_low_weight/track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                # f"solutions/wt_track_low_weight/track_mhe_wt_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
            )
            Nmhe = mat_content['N_mhe']
            N = mat_content['N_tot']
            NS = int(N - Nmhe)
            # Q_est_mean = np.mean(mat_content['X_est'][:, :biorbd_model.nbQ(), :], axis=[0, 1])
            # Qdot_est_mean = np.mean(mat_content['X_est'][:, biorbd_model.nbQ():biorbd_model.nbQ() * 2, :], axis=[0, 1])
            # A_est_mean = np.mean(mat_content['X_est'][:, -biorbd_model.nbMuscles():, :], axis=[0, 1])
            # U_est_mean = np.mean(mat_content['U_est'], axis=[0, 1])
            X_est = mat_content['X_est']
            U_est = mat_content['U_est']
            q_ref = mat_content['x_sol'][:biorbd_model.nbQ(), :NS + 1]
            dq_ref = mat_content['x_sol'][biorbd_model.nbQ():biorbd_model.nbQ() * 2, :NS + 1]
            a_ref = mat_content['x_sol'][-biorbd_model.nbMuscles():, :NS + 1]
            u_ref = mat_content['u_sol'][:, :NS]
            q_ref_try = np.ndarray((nb_try, biorbd_model.nbQ(), NS + 1))
            dq_ref_try = np.ndarray((nb_try, biorbd_model.nbQ(), NS + 1))
            a_ref_try = np.ndarray((nb_try, biorbd_model.nbMuscles(), NS + 1))
            u_ref_try = np.ndarray((nb_try, biorbd_model.nbMuscles(), NS))
            for i in range(nb_try):
                q_ref_try[i, :, :] = q_ref
                dq_ref_try[i, :, :] = dq_ref
                a_ref_try[i, :, :] = a_ref
                u_ref_try[i, :, :] = u_ref

            Q_err = np.linalg.norm(X_est[:, :biorbd_model.nbQ(), :] - q_ref_try, axis=2) / np.sqrt(NS + 1)
            Q_err = np.mean(Q_err, axis=1)
            Q_std = np.std(Q_err)
            Q_err = np.mean(Q_err)

            DQ_err = np.linalg.norm(
                X_est[:, biorbd_model.nbQ():biorbd_model.nbQ() * 2, :] - dq_ref_try, axis=2) / np.sqrt(NS + 1)
            DQ_err = np.mean(DQ_err, axis=1)
            DQ_std = np.std(DQ_err)
            DQ_err = np.mean(DQ_err)


            A_err = np.linalg.norm(
                X_est[:, -biorbd_model.nbMuscles():, :] - a_ref_try, axis=2) / np.sqrt(NS + 1)
            A_err = np.mean(A_err, axis=1)
            A_std = np.std(A_err)
            A_err = np.mean(A_err)

            U_err = np.linalg.norm(
                U_est[:, -biorbd_model.nbMuscles():, :] - u_ref_try, axis=2) / np.sqrt(NS)
            U_err = np.mean(U_err, axis=1)
            U_std = np.std(U_err)
            U_err = np.mean(U_err)


            markers_target_mean = np.mean(mat_content['markers_target'], axis=0)
            u_target_mean = np.mean(mat_content['u_target'], axis=0)
            Ns_mhe = mat_content['N_mhe']
            RMSE[co, marker_lvl, count] = Q_err
            STD[co, marker_lvl, count] = Q_std
            RMSE[co, marker_lvl, count + 1] = DQ_err
            STD[co, marker_lvl, count + 1] = DQ_std
            RMSE[co, marker_lvl, count + 2] = A_err
            STD[co, marker_lvl, count + 2] = A_std
            RMSE[co, marker_lvl, count + 3] = U_err
            STD[co, marker_lvl, count + 3] = U_std
            count += 4
form = '{0:.3f}'
for co in range(co_lvl):
    for marker_lvl in range(len(marker_noise_lvl)):
        print(
            f"{marker_lvl} & {float(form.format(RMSE[co, marker_lvl, 0]))} $ \pm $ {float(form.format(STD[co, marker_lvl, 0]))} "
            f"& {float(form.format(RMSE[co, marker_lvl, 1]))} $ \pm $ {float(form.format(STD[co, marker_lvl, 1]))} "
            f"& {float(form.format(RMSE[co, marker_lvl, 2]))} $ \pm $ {float(form.format(STD[co, marker_lvl, 2]))} "
            f"& {float(form.format(RMSE[co, marker_lvl, 3]))} $ \pm $ {float(form.format(STD[co, marker_lvl, 3]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 4]))} $ \pm $ {float(form.format(STD[co, marker_lvl, 4]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 5]))} $ \pm $ {float(form.format(STD[co, marker_lvl, 5]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 6]))} $ \pm $ {float(form.format(STD[co, marker_lvl, 6]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 7]))} $ \pm $ {float(form.format(STD[co, marker_lvl, 7]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 8]))} $ \pm $ {float(form.format(STD[co, marker_lvl, 8]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 9]))} $ \pm $ {float(form.format(STD[co, marker_lvl, 9]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 10]))} $ \pm $ {float(form.format(STD[co, marker_lvl,10]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 11]))} $ \pm $ {float(form.format(STD[co, marker_lvl,11]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 12]))} $ \pm $ {float(form.format(STD[co, marker_lvl,12]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 13]))} $ \pm $ {float(form.format(STD[co, marker_lvl,13]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 14]))} $ \pm $ {float(form.format(STD[co, marker_lvl,14]))} "
              f"& {float(form.format(RMSE[co, marker_lvl, 15]))} $ \pm $ {float(form.format(STD[co, marker_lvl,15]))} \\")


ax = seaborn.heatmap(RMSE[1, :, :], annot=True, linewidths=0.2, norm=LogNorm())
plt.show()
# for i in range(co_lvl):
#     pd.DataFrame(RMSE[i, :, :]).to_csv(f"solutions/RMSE_wt_EMG_{i}.csv")
#     # pd.DataFrame(RMSE[i, :, :]).to_csv(f"solutions/RMSE_w_EMG_{i}.csv")

# t = np.linspace(0, T, Ns + 1)
# q_name = [biorbd_model.nameDof()[i].to_string() for i in range(biorbd_model.nbQ())]
# plt.figure("Q")
# for i in range(biorbd_model.nbQ()):
#     plt.subplot(2, 3, i + 1)
#     # plt.plot(x_ref[i, :])
#     for j in range(nb_try):
#         plt.plot(X_est[j, i, :])
#         plt.title(q_name[i])
# plt.legend(range(nb_try), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#
# plt.figure("Q_dot")
# for i in range(biorbd_model.nbQ()):
#     plt.subplot(2, 3, i + 1)
#     plt.plot(x_ref[biorbd_model.nbQ() + i, :])
#     for j in range(nb_try):
#         plt.plot(X_est[j, biorbd_model.nbQ() + i, :])
#         plt.title(q_name[i])
# plt.legend(range(nb_try+1), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#
# # plt.figure("Tau")
# # for i in range(q.shape[0]):
# #     plt.subplot(2, 3, i + 1)
# #     plt.plot(t, tau[i, :], c='orange')
# #     plt.title(biorbd_model.muscleNames()[i].to_string())
#
# plt.figure("Muscles controls")
# for i in range(biorbd_model.nbMuscles()):
#     plt.subplot(4, 5, i + 1)
#     plt.plot(t, u_ref[i, :])
#     for j in range(nb_try):
#         plt.plot(U_est[j, i, :])
#         # plt.plot(t, a[i, :], c='purple')
#     plt.title(biorbd_model.muscleNames()[i].to_string())
# plt.legend(range(nb_try), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.show()

# b = bioviz.Viz(model_path="arm_wt_rot_scap.bioMod")
# b.load_movement(q_co[3, :, :])
# b.exec()