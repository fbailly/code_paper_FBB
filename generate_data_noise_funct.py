import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import biorbd
import pickle
from utils import *

T = 0.5
N = 150
# Ps = T/N
# Fs = 1/Ps
# biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
# with open(
#         f"solutions/sim_ac_{int(T * 1000)}ms_{N}sn_REACH2_co_level_0.bob", 'rb'
# ) as file:
#     data = pickle.load(file)
# states = data['data'][0]
# controls = data['data'][1]
# q_sol = states['q']
# dq_sol = states['q_dot']
# a_sol = states['muscles']
# u_sol = controls['muscles']
# u_co = u_sol
# xf = np.linspace(-Fs/2, Fs/2, N)
t = np.linspace(0, T, N)

def generate_noise(model, q, excitations, marker_noise_level, EMG_noise_level):
    biorbd_model = model
    q_sol = q
    u_co = excitations
    EMG_fft = scipy.fftpack.fft(u_co)
    EMG_no_noise = scipy.fftpack.ifft(EMG_fft)
    EMG_fft_noise = EMG_fft
    for k in range(biorbd_model.nbMuscles()):
        # EMG_fft_noise[k, 0] += np.random.normal(0, (np.real(EMG_fft_noise[k, 0]*0.2)))
        for i in range(1, 17, 3):
            if i in [4, 8]:
                rand_noise = np.random.normal(np.real(EMG_fft[k, i]) / i * EMG_noise_level,
                                              np.abs(np.real(EMG_fft[k, i]) * 0.2 * EMG_noise_level))

            elif i % 2 == 0:
                rand_noise = np.random.normal(2 * np.real(EMG_fft[k, i]) / i * EMG_noise_level,
                                              np.abs(np.real(EMG_fft[k, i]) * 0.2 * EMG_noise_level))

            else:
                rand_noise = np.random.normal(2 * np.real(EMG_fft[k, i]) / i * EMG_noise_level,
                                              np.abs(np.real(EMG_fft[k, i]) * EMG_noise_level * 5))
            EMG_fft_noise[k, i] += rand_noise
            EMG_fft_noise[k, -i] += rand_noise
    EMG_noise = np.real(scipy.fftpack.ifft(EMG_fft_noise))

    for i in range(biorbd_model.nbMuscles()):
        for j in range(EMG_noise.shape[1]):
            if EMG_noise[i, j] < 0:
                EMG_noise[i, j] = 0

    # plt.figure("Muscles controls")
    # for i in range(biorbd_model.nbMuscles()):
    #     plt.subplot(4, 5, i + 1)
    #     plt.plot(np.real(EMG_no_noise[i, :]))
    #     plt.plot(np.real(EMG_noise[i, :]))
    #     plt.plot(u_co[i, :])
    #     plt.title(biorbd_model.muscleNames()[i].to_string())
    # plt.legend(labels=['without_noise', 'with_noise', 'ref'], bbox_to_anchor=(1.05, 1), loc='upper left',
    #            borderaxespad=0.)
    # plt.show()

    # Ref
    n_mark = biorbd_model.nbMarkers()
    # get_markers = markers_fun(biorbd_model)
    # markers = np.zeros((3, biorbd_model.nbMarkers(), q_sol.shape[1]))
    # for i in range(q_sol.shape[1]):
    #     markers[:, :, i] = get_markers(q_sol[:, i])

    for i in range(n_mark):
        noise_position = MX(np.random.normal(0, marker_noise_level, 3)) + biorbd_model.marker(i).to_mx()
        biorbd_model.marker(i).setPosition(biorbd.Vector3d(noise_position[0], noise_position[1], noise_position[2]))

    get_markers = markers_fun(biorbd_model)
    markers_target_noise = np.zeros((3, biorbd_model.nbMarkers(), q_sol.shape[1]))
    for i in range(q_sol.shape[1]):
        markers_target_noise[:, :, i] = get_markers(q_sol[:, i])


    # plt.figure("Markers")
    # for i in range(markers_target_noise.shape[1]):
    #     plt.plot(np.linspace(0, 1, markers_target_noise.shape[2]), markers_target_noise[:, i, :].T, "k")
    #     # plt.plot(np.linspace(0, 1, markers.shape[2]), markers[:, i, :].T, "r--")
    # plt.xlabel("Time")
    # plt.ylabel("Markers Position")
    # plt.show()

    return markers_target_noise, EMG_noise




