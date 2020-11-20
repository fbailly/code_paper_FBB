import biorbd
import seaborn
import numpy as np
from casadi import MX, Function
import matplotlib.pyplot as plt
import pickle
from generate_data_noise_funct import generate_noise

def markers_fun(biorbd_model):
    qMX = MX.sym('qMX', biorbd_model.nbQ())
    return Function('markers', [qMX], [biorbd_model.markers(qMX)])

biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
# Variable of the problem
Ns = 100
T = 0.8

tau_init = 0
muscle_init = 0.5
nbMT = biorbd_model.nbMuscleTotal()
nbQ = biorbd_model.nbQ()
nbGT = 0
# Define noise
marker_noise_lvl = [0, 0.002, 0.005, 0.01]
EMG_noise_lvl = [0, 0.05, 0.1, 0.2]


nb_co_lvl = 4
EMG_noise = np.ndarray((len(EMG_noise_lvl), nbMT, Ns+1))
co = '0'
motion = 'REACH2'
with open(
        f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{co}.bob", 'rb'
) as file:
    data = pickle.load(file)
states = data['data'][0]
controls = data['data'][1]
q_sol = states['q']
dq_sol = states['q_dot']
a_sol = states['muscles']
u_sol = controls['muscles']
w_tau = 'tau' in controls.keys()
if w_tau:
    tau = controls['tau']
else:
    tau = np.zeros((nbGT, Ns + 1))
# Loop for marker and EMG noise
for marker_lvl in range(1):
    for EMG_lvl in range(len(EMG_noise_lvl)):
        get_markers = markers_fun(biorbd_model)
        markers_target = np.zeros((3, biorbd_model.nbMarkers(), Ns + 1))
        for i in range(Ns + 1):
            markers_target[:, :, i] = get_markers(q_sol[:, i])
        EMG_noise[EMG_lvl, :, :] = u_sol

        x_sol = np.concatenate((q_sol, dq_sol, a_sol))
        if marker_lvl != 0:
            markers_target = generate_noise(
                biorbd_model, q_sol, u_sol, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
            )[0]

        if EMG_lvl != 0:
            EMG_noise[EMG_lvl, :, :] = generate_noise(
                biorbd_model, q_sol, u_sol, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
            )[1]

t = np.linspace(0, T, Ns+1)
seaborn.set_style("whitegrid")
seaborn.color_palette()

fig = plt.figure("Muscles controls")
plt.gcf().subplots_adjust(left=0.06, right=0.99, wspace=0.25, hspace=0.2)

for i in range(biorbd_model.nbMuscles()):
    fig = plt.subplot(4, 5, i + 1)
    if i in [14, 15, 16, 17, 18]:
        plt.xlabel('Time(s)')

    else:
        fig.set_xticklabels([])
    if i in [0, 5, 10, 15]:
        plt.ylabel('Simulated EMG')

    if i == 18:
        for k in range(len(EMG_noise_lvl)):
            plt.plot(t, np.real(EMG_noise[k, i, :]), label=f'EMG noise level {k}')
            # plt.plot(t, u_sol[i, :])
    else:
        for k in range(len(EMG_noise_lvl)):
            plt.plot(t, np.real(EMG_noise[k, i, :]), label=f'EMG noise level {k}')
    plt.title(biorbd_model.muscleNames()[i].to_string())
plt.legend(bbox_to_anchor=(1.05, 0.80),loc='upper left', frameon=False)
plt.show()