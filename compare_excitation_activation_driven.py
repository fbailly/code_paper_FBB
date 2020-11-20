import numpy as np
import biorbd
import pickle
import scipy.io as sio
from casadi import MX, Function, vertcat, jacobian
from matplotlib import pyplot as plt


use_torque = False
T = 0.5
Ns = 150
motion = 'REACH2'
i = '0'
biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
with open(
        f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{i}.bob", 'rb'
) as file:
    data = pickle.load(file)
states = data['data'][0]
controls = data['data'][1]
q_sol = states['q']
dq_sol = states['q_dot']
a_sol = states['muscles']
u_sol = controls['muscles']
with open(
        f"solutions/tracking_markers_EMG_activations_driven.bob", "rb"
) as file:
    data = pickle.load(file)
states = data['data'][0]
controls = data['data'][1]
q_act = states['q']
dq_act = states['q_dot']
a_act = controls['muscles']

with open(
        f"solutions/tracking_markers_EMG_excitations_driven.bob", "rb"
) as file:
    data = pickle.load(file)
states = data['data'][0]
controls = data['data'][1]
q_exc = states['q']
dq_exc = states['q_dot']
a_exc = states['muscles']
u_exc = controls['muscles']

plt.figure('Muscles activations')
for i in range(biorbd_model.nbMuscles()):
    plt.subplot(4, 5, i + 1)
    plt.plot(a_act[i, :])
    plt.plot(a_exc[i, :])
    plt.plot(a_sol[i, :], 'r--')
    plt.title(biorbd_model.muscleNames()[i].to_string())
plt.legend(
    labels=['a_activation_driven', 'a_excitation_driven', 'a_ref'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.
)
plt.figure('Muscles excitations')
for i in range(biorbd_model.nbMuscles()):
    plt.subplot(4, 5, i + 1)
    plt.plot(a_act[i, :])
    plt.plot(u_exc[i, :])
    plt.plot(u_sol[i, :], 'r--')
    plt.title(biorbd_model.muscleNames()[i].to_string())
plt.legend(
    labels=['a_activation_driven', 'u_excitation_driven', 'u_ref'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.
)
plt.figure('RMSE_activations')
RMSE_act = np.sqrt(np.sum(((a_sol - a_act)**2), axis=1)/Ns)
RMSE_exc = np.sqrt(np.sum(((a_sol - a_exc)**2), axis=1)/Ns)
x1 = range(biorbd_model.nbMuscles())
x2 = [i + 0.2 for i in x1]
width = 0.2
plt.bar(x1, RMSE_act, width, color='red')
plt.bar(x2, RMSE_exc, width, color='blue')
plt.legend(labels=['activation_driven', 'excitation_driven'])
plt.xlabel('muscles')
plt.ylabel('RMSE')
plt.figure('RMSE_excitations')
RMSE_act = np.sqrt(np.sum(((u_sol - a_act)**2), axis=1)/Ns)
RMSE_exc = np.sqrt(np.sum(((u_sol - u_exc)**2), axis=1)/Ns)
x1 = range(biorbd_model.nbMuscles())
x2 = [i + 0.2 for i in x1]
width = 0.2
plt.bar(x1, RMSE_act, width, color='red')
plt.bar(x2, RMSE_exc, width, color='blue')
plt.legend(labels=['activation_driven', 'excitation_driven'])
plt.xlabel('muscles')
plt.ylabel('RMSE')
plt.show()