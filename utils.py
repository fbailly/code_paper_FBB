import numpy as np
from casadi import MX, Function, horzcat
from math import *
import biorbd
import csv
import warnings
import scipy
import scipy.fftpack
import pickle
import matplotlib.pyplot as plt
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    Data,
    InterpolationType,
    Bounds,
)


# Use biorbd function for inverse kinematics
def markers_fun(biorbd_model):
    qMX = MX.sym("qMX", biorbd_model.nbQ())
    return Function(
        "markers", [qMX], [horzcat(*[biorbd_model.markers(qMX)[i].to_mx() for i in range(biorbd_model.nbMarkers())])]
    )


# Return muscle force
def muscles_forces(q, qdot, act, controls, model, use_activation=False):
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscles())
    for k in range(model.nbMuscles()):
        if use_activation:
            muscles_states[k].setActivation(controls[k])
        else:
            muscles_states[k].setExcitation(controls[k])
            muscles_states[k].setActivation(act[k])
    muscles_force = model.muscleForces(muscles_states, q, qdot).to_mx()
    return muscles_force


# Return biorbd muscles force function
def force_func(biorbd_model, use_activation=False):
    qMX = MX.sym("qMX", biorbd_model.nbQ(), 1)
    dqMX = MX.sym("dqMX", biorbd_model.nbQ(), 1)
    aMX = MX.sym("aMX", biorbd_model.nbMuscles(), 1)
    uMX = MX.sym("uMX", biorbd_model.nbMuscles(), 1)
    return Function(
        "MuscleForce",
        [qMX, dqMX, aMX, uMX],
        [muscles_forces(qMX, dqMX, aMX, uMX, biorbd_model, use_activation=use_activation)],
        ["qMX", "dqMX", "aMX", "uMX"],
        ["Force"],
    ).expand()


# Return mean RMSE
def compute_err(
    init_offset,
    final_offset,
    Ns_mhe,
    X_est,
    U_est,
    Ns,
    model,
    q,
    dq,
    tau,
    activations,
    excitations,
    nbGT,
    ratio=1,
    use_activation=False,
    full_windows=False
):
    # All variables
    model = model
    get_force = force_func(model, use_activation=use_activation)
    get_markers = markers_fun(model)
    err = dict()
    offset = final_offset - Ns_mhe
    q_ref = q[:, 0 : Ns + 1 : ratio]
    dq_ref = dq[:, 0 : Ns + 1 : ratio]
    tau_ref = tau[:, 0:Ns:ratio]
    if use_activation:
        muscles_ref = activations[:, 0:Ns:ratio]
    else:
        muscles_ref = excitations[nbGT:, 0:Ns:ratio]
    sol_mark = np.zeros((3, model.nbMarkers(), ceil((Ns + 1) / ratio) - Ns_mhe))
    sol_mark_ref = np.zeros((3, model.nbMarkers(), ceil((Ns + 1) / ratio) - Ns_mhe))

    # Compute RMSE
    err["q"] = np.sqrt(
        np.square(X_est[: model.nbQ(), init_offset:-offset] - q_ref[:, init_offset:-final_offset]).mean(axis=1)
    ).mean()
    err["q_dot"] = np.sqrt(
        np.square(
            X_est[model.nbQ() : model.nbQ() * 2, init_offset:-offset] - dq_ref[:, init_offset:-final_offset]
        ).mean(axis=1)
    ).mean()
    err["tau"] = np.sqrt(
        np.square(U_est[:nbGT, init_offset:-offset] - tau_ref[:nbGT, init_offset:-final_offset]).mean(axis=1)
    ).mean()
    err["muscles"] = np.sqrt(
        np.square(U_est[nbGT:, init_offset:-offset] - muscles_ref[:, init_offset:-final_offset]).mean(axis=1)
    ).mean()

    # Get marker and compute RMSE
    len_x = int(ceil((Ns + 1) / ratio) - Ns_mhe) if full_windows is not True else Ns + 1
    len_u = int(ceil(Ns / ratio) - Ns_mhe) if full_windows is not True else Ns
    for i in range(len_x):
        sol_mark[:, :, i] = get_markers(X_est[: model.nbQ(), i])
    sol_mark_tmp = np.zeros((3, sol_mark_ref.shape[1], Ns + 1))
    for i in range(Ns + 1):
        sol_mark_tmp[:, :, i] = get_markers(q[:, i])
    sol_mark_ref = sol_mark_tmp[:, :, 0 : Ns + 1 : ratio]
    err["markers"] = np.sqrt(
        np.square(sol_mark[:, :, init_offset:-offset] - sol_mark_ref[:, :, init_offset:-final_offset])
        .sum(axis=0)
        .mean(axis=1)
    ).mean()

    # Get muscle force and compute RMSE
    force_ref_tmp = np.ndarray((model.nbMuscles(), Ns))
    force_est = np.ndarray((model.nbMuscles(), len_u))
    if use_activation:
        a_est = np.zeros((model.nbMuscles(), Ns))
    else:
        a_est = X_est[-model.nbMuscles() :, :]

    for i in range(model.nbMuscles()):
        for j in range(len_u):
            force_est[i, j] = get_force(
                X_est[: model.nbQ(), j], X_est[model.nbQ() : model.nbQ() * 2, j], a_est[:, j], U_est[nbGT:, j]
            )[i, :]
    get_force = force_func(model, use_activation=False)
    for i in range(model.nbMuscles()):
        for k in range(Ns):
            force_ref_tmp[i, k] = get_force(q[:, k], dq[:, k], activations[:, k], excitations[nbGT:, k])[i, :]
    force_ref = force_ref_tmp[:, 0:Ns:ratio]
    err["force"] = np.sqrt(
        np.square(force_est[:, init_offset:-offset] - force_ref[:, init_offset:-final_offset]).mean(axis=1)
    ).mean()

    return err


# Return estimation on the first node of the ocp and inital guess for next optimisation
def warm_start_mhe(ocp, sol, use_activation=False):
    # Define problem variable
    states, controls = Data.get_data(ocp, sol)
    q, qdot = states["q"], states["qdot"]
    u = controls["muscles"]
    x = np.vstack([q, qdot]) if use_activation else np.vstack([q, qdot, states["muscles"]])
    w_tau = "tau" in controls.keys()
    if w_tau:
        u = np.vstack([controls["tau"], u])
    else:
        u = u
    # Prepare data to return
    x0 = np.hstack((x[:, 1:], np.tile(x[:, [-1]], 1)))  # discard oldest estimate of the window, duplicates youngest
    u0 = u[:, 1:]
    x_out = x[:, 0]
    u_out = u[:, 0]
    return x0, u0, x_out, u_out


# Return which iteration has not converged
def convert_txt_output_to_list(file, nbco, nbmark, nbemg, nbtries):
    conv_list = [[[[[] for i in range(nbtries)] for j in range(nbemg)] for k in range(nbmark)] for l in range(nbco)]
    with open(file) as f:
        fdel = csv.reader(f, delimiter=";", lineterminator="\n")
        for line in fdel:
            if line[0] == "7":
                try:
                    conv_list[int(line[1])][int(line[2])][int(line[3])][int(line[4])].append(line[5])
                except:
                    warnings.warn(f"line {line} ignored")
    return conv_list


# Return noise on EMG and Markers
def generate_noise(model, q, excitations, marker_noise_level, EMG_noise_level):
    biorbd_model = model
    q_sol = q
    u_co = excitations

    # Noise on EMG using Fast Fourier transformation
    EMG_fft = scipy.fftpack.fft(u_co)
    EMG_fft_noise = EMG_fft
    for k in range(biorbd_model.nbMuscles()):
        # EMG_fft_noise[k, 0] += np.random.normal(0, (np.real(EMG_fft_noise[k, 0]*0.2)))
        for i in range(1, 17, 3):
            if i in [4, 8]:
                rand_noise = np.random.normal(
                    np.real(EMG_fft[k, i]) / i * EMG_noise_level, np.abs(np.real(EMG_fft[k, i]) * 0.2 * EMG_noise_level)
                )

            elif i % 2 == 0:
                rand_noise = np.random.normal(
                    2 * np.real(EMG_fft[k, i]) / i * EMG_noise_level,
                    np.abs(np.real(EMG_fft[k, i]) * 0.2 * EMG_noise_level),
                )

            else:
                rand_noise = np.random.normal(
                    2 * np.real(EMG_fft[k, i]) / i * EMG_noise_level,
                    np.abs(np.real(EMG_fft[k, i]) * EMG_noise_level * 5),
                )
            EMG_fft_noise[k, i] += rand_noise
            EMG_fft_noise[k, -i] += rand_noise
    EMG_noise = np.real(scipy.fftpack.ifft(EMG_fft_noise))
    for i in range(biorbd_model.nbMuscles()):
        for j in range(EMG_noise.shape[1]):
            if EMG_noise[i, j] < 0:
                EMG_noise[i, j] = 0

    # Noise on marker position with gaussian normal distribution
    n_mark = biorbd_model.nbMarkers()
    for i in range(n_mark):
        noise_position = MX(np.random.normal(0, marker_noise_level, 3)) + biorbd_model.marker(i).to_mx()
        biorbd_model.marker(i).setPosition(biorbd.Vector3d(noise_position[0], noise_position[1], noise_position[2]))
    get_markers = markers_fun(biorbd_model)
    markers_target_noise = np.zeros((3, biorbd_model.nbMarkers(), q_sol.shape[1]))
    for i in range(q_sol.shape[1]):
        markers_target_noise[:, :, i] = get_markers(q_sol[:, i])

    return markers_target_noise, EMG_noise


# Return states and controls on the last node to keep the continuity of the problem
def switch_phase(ocp, sol):
    data = Data.get_data(ocp, sol)
    q = data[0]["q"]
    dq = data[0]["qdot"]
    act = data[0]["muscles"]
    exc = data[1]["muscles"]
    x = np.vstack([q, dq, act])
    return x[:, :-1], exc[:, :-1], x[:, -1]


def define_objective(
    use_activation,
    use_torque,
    TRACK_EMG,
    iter,
    rt_ratio,
    nbGT,
    Ns_mhe,
    muscles_target,
    markers_target,
    with_low_weight,
    biorbd_model,
    idx,
    TRACK_LESS_EMG=False,
    full_windows=False
):
    objectives = ObjectiveList()
    idx = idx if TRACK_LESS_EMG else range(biorbd_model.nbMuscles())
    if TRACK_EMG:
        w_marker = 10000000 if with_low_weight else 1000000000
        w_control = 1000000 if with_low_weight else 1000000
        if full_windows:
            w_marker = 1000000 if with_low_weight else 100000000
            w_control = 10000 if with_low_weight else 100000
        w_torque = 100000000 if full_windows else 100000000
        objectives.add(
            ObjectiveFcn.Lagrange.TRACK_MUSCLES_CONTROL,
            weight=w_control,
            target=muscles_target[idx, iter * rt_ratio : (Ns_mhe + iter) * rt_ratio : rt_ratio],
            index=idx,
        )
        if TRACK_LESS_EMG:
            objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10000)
    else:
        if use_activation:
            w_control = 100000 if full_windows else 100000
            if full_windows:
                w_marker = 1000000 if with_low_weight else 100000000
            else:
                w_marker = 10000000 if with_low_weight else 10000000
        else:
            w_control = 100000 if full_windows else 1000000
            if full_windows:
                w_marker = 1000000 if with_low_weight else 1000000
            else:
                w_marker = 10000000 if with_low_weight else 100000000

        w_torque = 100000000 if full_windows else 10000000
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=w_control)

    if use_torque:
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=w_torque)
    objectives.add(
        ObjectiveFcn.Lagrange.TRACK_MARKERS,
        weight=w_marker,
        target=markers_target[:, :, iter * rt_ratio : (Ns_mhe + 1 + iter) * rt_ratio : rt_ratio],
    )
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=10, index=np.array(range(biorbd_model.nbQ())))
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=10,
        index=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)),
    )
    if use_activation is not True:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            weight=10,
            index=np.array(range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles())),
        )
    return objectives


def get_reference_movement(file, use_torque, nbGT, Ns):
    with open(file, "rb") as file:
        data = pickle.load(file)
    states = data["data"][0]
    controls = data["data"][1]
    q_ref = states["q"]
    dq_ref = states["q_dot"]
    a_ref = states["muscles"]
    w_tau = "tau" in controls.keys()  # Check if there are residuals torques
    tau = controls["tau"] if w_tau else np.zeros((nbGT, Ns + 1))
    if use_torque:
        u_ref = np.concatenate((tau, controls["muscles"]))
    else:
        u_ref = controls["muscles"]
    return np.concatenate((q_ref, dq_ref, a_ref)), u_ref


def plot_results(
    biorbd_model,
    X_est,
    q_ref,
    Ns,
    rt_ratio,
    nbQ,
    dq_ref,
    U_est,
    u_ref,
    nbGT,
    muscles_target,
    force_est,
    force_ref,
    tries,
    markers_target,
    use_torque,
):
    plt.figure("q")
    for i in range(biorbd_model.nbQ()):
        plt.subplot(3, 2, i + 1)
        plt.plot(X_est[i, :], "x")
        plt.plot(q_ref[i, 0 : Ns + 1 : rt_ratio])
    plt.legend(labels=["q_est", "q_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    plt.figure("qdot")
    for i in range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2):
        plt.subplot(3, 2, i - nbQ + 1)
        plt.plot(X_est[i, :], "x")
        plt.plot(dq_ref[i - nbQ, 0 : Ns + 1 : rt_ratio])
    plt.legend(labels=["q_est", "q_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    if use_torque:
        plt.figure("Tau")
        for i in range(biorbd_model.nbQ()):
            plt.subplot(3, 2, i + 1)
            plt.plot(U_est[i, :], "x")
            plt.plot(u_ref[i, 0 : Ns + 1 : rt_ratio])
            plt.plot(muscles_target[i, :], "k--")
        plt.legend(labels=["Tau_est", "Tau_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    plt.figure("Muscles excitations")
    for i in range(biorbd_model.nbMuscles()):
        plt.subplot(4, 5, i + 1)
        plt.plot(U_est[nbGT + i, :])
        plt.plot(u_ref[nbGT + i, 0:Ns:rt_ratio], c="red")
        plt.plot(muscles_target[nbGT + i, 0:Ns:rt_ratio], "k--")
        plt.title(biorbd_model.muscleNames()[i].to_string())
    plt.legend(
        labels=["u_est", "u_ref", "u_with_noise"],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.figure("Muscles_force")
    for i in range(biorbd_model.nbMuscles()):
        plt.subplot(4, 5, i + 1)
        plt.plot(force_est[tries, i, :])
        plt.plot(force_ref[i, 0:Ns:rt_ratio], c="red")
        plt.title(biorbd_model.muscleNames()[i].to_string())
    plt.legend(labels=["f_est", "f_ref"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    get_markers = markers_fun(biorbd_model)
    markers = np.zeros((3, biorbd_model.nbMarkers(), q_ref.shape[1]))
    for i in range(q_ref.shape[1]):
        markers[:, :, i] = get_markers(q_ref[:, i])
    markers_est = np.zeros((3, biorbd_model.nbMarkers(), X_est.shape[1]))
    for i in range(X_est.shape[1]):
        markers_est[:, :, i] = get_markers(X_est[: biorbd_model.nbQ(), i])

    plt.figure("Markers")
    for i in range(markers_target.shape[1]):
        plt.plot(markers_target[:, i, 0:Ns:rt_ratio].T, "k")
        plt.plot(markers[:, i, 0:Ns:rt_ratio].T, "r--")
        plt.plot(markers_est[:, i, :].T, "b")
    plt.xlabel("Time")
    plt.ylabel("Markers Position")

    return plt.show()


def prepare_ocp(
    biorbd_model,
    final_time,
    x0,
    nbGT,
    number_shooting_points,
    use_SX=False,
    nb_threads=8,
    use_torque=False,
    use_activation=False,
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd_model
    nbGT = nbGT
    nbMT = biorbd_model.nbMuscleTotal()
    tau_min, tau_max, tau_init = -100, 100, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    activation_min, activation_max, activation_init = 0, 1, 0.2

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsList()
    if use_activation and use_torque:
        dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)
    elif use_activation is not True and use_torque:
        dynamics.add(DynamicsFcn.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN)
    elif use_activation and use_torque is not True:
        dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_DRIVEN)
    elif use_activation is not True and use_torque is not True:
        dynamics.add(DynamicsFcn.MUSCLE_EXCITATIONS_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    if use_activation is not True:
        x_bounds[0].concatenate(
            Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
        )

    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * nbGT + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [tau_max] * nbGT + [muscle_max] * biorbd_model.nbMuscleTotal(),
    )

    # Initial guesses
    x_init = InitialGuess(np.tile(x0, (number_shooting_points + 1, 1)).T, interpolation=InterpolationType.EACH_FRAME)

    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)
    u_init = InitialGuess(np.tile(u0, (number_shooting_points, 1)).T, interpolation=InterpolationType.EACH_FRAME)
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        use_sx=use_SX,
        n_threads=nb_threads,
    )
