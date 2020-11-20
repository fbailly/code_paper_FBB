import numpy as np
from casadi import MX, Function, horzcat
from math import *
from bioptim import Data
import biorbd
import csv

def markers_fun(biorbd_model):
    qMX = MX.sym('qMX', biorbd_model.nbQ())
    return Function('markers', [qMX],
                    [horzcat(*[biorbd_model.markers(qMX)[i].to_mx() for i in range(biorbd_model.nbMarkers())])])


def compute_err_mhe(init_offset, final_offset, Ns_mhe, X_est, U_est, Ns, model, q, dq, tau,
                activations, excitations, nbGT, ratio=1, use_activation=False):
    model = model
    get_markers = markers_fun(model)
    err = dict()
    offset = final_offset - Ns_mhe
    nbGT = nbGT
    Ns = Ns
    q_ref = q[:, 0:Ns + 1:ratio]
    dq_ref = dq[:, 0:Ns + 1:ratio]
    tau_ref = tau[:, 0:Ns:ratio]
    muscles_ref = excitations[:, 0:Ns:ratio]
    if use_activation:
        muscles_ref = activations[:, 0:Ns:ratio]
    sol_mark = np.zeros((3, model.nbMarkers(), ceil((Ns + 1) / ratio) - Ns_mhe))
    sol_mark_ref = np.zeros((3, model.nbMarkers(), ceil((Ns + 1) / ratio) - Ns_mhe))
    err['q'] = np.sqrt(np.square(X_est[:model.nbQ(), init_offset:-offset] - q_ref[:, init_offset:-final_offset]).mean(axis=1)).mean()
    err['q_dot'] = np.sqrt(
        np.square(X_est[model.nbQ():model.nbQ() * 2, init_offset:-offset] - dq_ref[:, init_offset:-final_offset]).mean(axis=1)).mean()
    err['tau'] = np.sqrt(np.square(U_est[:nbGT, init_offset:-offset] - tau_ref[:, init_offset:-final_offset]).mean(axis=1)).mean()
    err['muscles'] = np.sqrt(np.square(U_est[nbGT:, init_offset:-offset] - muscles_ref[:, init_offset:-final_offset]).mean(axis=1)).mean()
    for i in range(ceil((Ns + 1) / ratio) - Ns_mhe):
        sol_mark[:, :, i] = get_markers(X_est[:model.nbQ(), i])
    sol_mark_tmp = np.zeros((3, sol_mark_ref.shape[1], Ns + 1))
    for i in range(Ns + 1):
        sol_mark_tmp[:, :, i] = get_markers(q[:, i])
    sol_mark_ref = sol_mark_tmp[:, :, 0:Ns + 1:ratio]
    err['markers'] = np.sqrt(np.square(sol_mark[:, :, init_offset:-offset] - sol_mark_ref[:, :, init_offset:-final_offset]).mean(axis=1)).mean()
    return err

def compute_err(init_offset, final_offset, X_est, U_est, Ns, model, q, dq, tau,
                activations, excitations, nbGT, use_activation=False):
    model = model
    get_markers = markers_fun(model)
    err = dict()
    nbGT = nbGT
    Ns = Ns
    q_ref = q[:, 0:Ns + 1]
    dq_ref = dq[:, 0:Ns + 1]
    tau_ref = tau[:, 0:Ns]
    muscles_ref = excitations[:, 0:Ns]
    if use_activation:
        muscles_ref = activations[:, 0:Ns]
    sol_mark = np.zeros((3, model.nbMarkers(), Ns + 1))
    sol_mark_ref = np.zeros((3, model.nbMarkers(), Ns + 1))
    err['q'] = np.sqrt(np.square(X_est[:model.nbQ(), init_offset:-final_offset] - q_ref[:, init_offset:-final_offset]).mean(axis=1)).mean()
    err['q_dot'] = np.sqrt(
        np.square(X_est[model.nbQ():model.nbQ() * 2, init_offset:-final_offset] - dq_ref[:, init_offset:-final_offset]).mean(axis=1)).mean()
    err['tau'] = np.sqrt(np.square(U_est[:nbGT, init_offset:-final_offset-1] - tau_ref[:, init_offset:-final_offset]).mean(axis=1)).mean()
    err['muscles'] = np.sqrt(np.square(U_est[nbGT:, init_offset:-final_offset-1] - muscles_ref[:, init_offset:-final_offset]).mean(axis=1)).mean()
    for i in range(Ns + 1):
        sol_mark[:, :, i] = get_markers(X_est[:model.nbQ(), i])
    sol_mark_tmp = np.zeros((3, sol_mark_ref.shape[1], Ns + 1))
    for i in range(Ns + 1):
        sol_mark_tmp[:, :, i] = get_markers(q[:, i])
    sol_mark_ref = sol_mark_tmp[:, :, 0:Ns + 1]
    err['markers'] = np.sqrt(np.square(sol_mark[:, :, init_offset:-final_offset] - sol_mark_ref[:, :, init_offset:-final_offset]).mean(axis=1)).mean()
    return err

def warm_start_mhe(ocp, sol, use_activation=False):
    data = Data.get_data(ocp, sol)
    q = data[0]["q"]
    dq = data[0]["q_dot"]
    tau = []
    if use_activation:
        act = data[1]["muscles"]
        x = np.vstack([q, dq])
        u = act
    else:
        act = data[0]["muscles"]
        exc = data[1]["muscles"]
        x = np.vstack([q, dq, act])
        u = exc
    w_tau = 'tau' in data[1].keys()
    if w_tau:
        tau = data[1]["tau"]
        u = np.vstack([tau, act])
    x0 = np.hstack((x[:, 1:], np.tile(x[:, [-1]], 1)))  # discard oldest estimate of the window, duplicates youngest
    u0 = u[:, 1:]  # discard oldest estimate of the window
    x_out = x[:, 0]
    u_out = u[:, 0]
    return x0, u0, x_out, u_out


def get_MHE_time_lenght(Ns_mhe, use_activation=False):
    # Nmhe>2
    # To be adjusted to guarantee real-time
    # Based on frequencies extracted from Fig.1
    if use_activation is not True:
        times_lenght = [0.024, 0.024, 0.024, 0.024,  # 1 sample on 3
                        0.032, 0.032,  # 1 sample on 4
                        0.04, 0.044,  # 1 sample on 5
                        0.048, 0.048, 0.048, 0.048, 0.048, 0.048,  # 1 sample on 6
                        0.056,  # 1 sample on 7
                        0.064, 0.064,  # 1 sample on 8
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    return times_lenght[Ns_mhe-2]

def muscles_forces(q, qdot, act, controls, model):
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscles())
    for k in range(model.nbMuscles()):
        muscles_states[k].setExcitation(controls[k])
        muscles_states[k].setActivation(act[k])
    # muscles_tau = model.muscularJointTorque(muscles_states, True,  q, qdot).to_mx()
    muscles_force = model.muscleForces(muscles_states, q, qdot).to_mx()
    return muscles_force

def convert_txt_output_to_list(file, nbco, nbmark, nbemg, nbtries):
    conv_list = [[[[[] for i in range(nbtries)] for j in range(nbemg)] for k in range(nbmark)] for l in range(nbco)]
    with open(file) as f:
        fdel = csv.reader(f, delimiter=';', lineterminator='\n')
        for line in fdel:
            if line[0] == '7':
                conv_list[int(line[1])][int(line[2])][int(line[3])][int(line[4])].append(line[5])
    return conv_list