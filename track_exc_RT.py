import biorbd
from time import time
import numpy as np
from casadi import MX, Function, horzcat
from math import *
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
import sys
from generate_data_noise_funct import generate_noise
import os
from utils import *

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    Data,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialGuessOption,
    Solver,
    InterpolationType,
    Bounds,
)

def prepare_ocp(
        biorbd_model,
        final_time,
        x0,
        nbGT,
        number_shooting_points,
        use_SX=False,
        nb_threads=8,
        use_torque=True
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd_model
    nbQ = biorbd_model.nbQ()
    nbGT = nbGT
    nbMT = biorbd_model.nbMuscleTotal()
    tau_min, tau_max, tau_init = -100, 100, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    activation_min, activation_max, activation_init = 0, 1, 0.2

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsTypeList()
    if use_activation and use_torque:
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)
    elif use_activation is not True and use_torque:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN)
    elif use_activation and use_torque is not True:
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_DRIVEN)
    elif use_activation is not True and use_torque is not True:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    if use_activation is not True:
        x_bounds[0].concatenate(
            Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
        )

    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [tau_min] * nbGT + [muscle_min] * biorbd_model.nbMuscleTotal(),
            [tau_max] * nbGT + [muscle_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    # Initial guesses
    x_init = InitialGuessOption(np.tile(x0, (number_shooting_points+1, 1)).T,
                                interpolation=InterpolationType.EACH_FRAME)

    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)+0.1
    u_init = InitialGuessOption(np.tile(u0, (number_shooting_points, 1)).T,
                                interpolation=InterpolationType.EACH_FRAME)
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
        use_SX=use_SX,
        nb_threads=nb_threads,
    )


if __name__ == "__main__":
    # Configuration of the problem
    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
    use_torque = False
    use_activation = False
    use_ACADOS = True
    WRITE_STATS = False
    save_results = True
    TRACK_EMG = True
    if TRACK_EMG:
        fold = "solutions/w_track_emg_rt/"
    else:
        fold = "solutions/wt_track_emg_rt/"
    use_noise = True
    use_co = True
    use_bash = False
    use_try = True
    use_N_elec = False
    if use_activation:
        use_N_elec = True
    # Variable of the problem
    T = 8
    Ns = 800
    T_elec = 0.02
    N_elec = int(T_elec * Ns / T)
    final_offset = 15
    init_offset = 15

    Ns_mhe = 7
    if use_bash:
        Ns_mhe = int(sys.argv[1])

    if use_activation is not True:
        rt_ratio_tot = [2, 2,
                        3, 3, 3, 3,
                        4, 4,
                        5, 5, 5, 5, 5, 5,
                        7, 7, 7, 7,
                        8, 8, 8]
    else:
        rt_ratio_tot = [2, 2, 2, 2, 2, 2,
                        3, 3, 3, 3, 3, 3,
                        4, 4, 4, 4, 4, 4, 4,
                        5, 5, 5, 5,
                        6, 6, 6, 6, 6, 6, 6]

    rt_ratio = rt_ratio_tot[Ns_mhe-2]
    T_mhe = T / (Ns / rt_ratio) * Ns_mhe

    if use_try:
        nb_try = 30
    else:
        nb_try = 1

    tau_init = 0
    muscle_init = 0.5
    nbMT = biorbd_model.nbMuscleTotal()
    nbQ = biorbd_model.nbQ()
    if use_torque:
        nbGT = biorbd_model.nbGeneralizedTorque()
    else:
        nbGT = 0
    if use_activation:
        x_ref = np.zeros((biorbd_model.nbQ()*2))
    else:
        x_ref = np.zeros((biorbd_model.nbQ()*2 + biorbd_model.nbMuscles()))

    # Define noise
    if use_noise:
        marker_noise_lvl = [0, 0.002, 0.005, 0.01]
        EMG_noise_lvl = [0, 0.6, 1, 1.5]
    else:
        marker_noise_lvl = [0]
        EMG_noise_lvl = [0]

    # define x_est u_est size
    if use_activation:
        X_est = np.zeros((biorbd_model.nbQ() * 2, ceil((Ns+1)/rt_ratio) - Ns_mhe))
    else:
        X_est = np.zeros((biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), ceil((Ns+1)/rt_ratio) - Ns_mhe))
    U_est = np.zeros((nbGT + biorbd_model.nbMuscleTotal(), ceil(Ns/rt_ratio) - Ns_mhe))

    # Set number of co-contraction level
    nb_co_lvl = 1
    if use_co:
        nb_co_lvl = 4
    qMX = MX.sym("qMX", biorbd_model.nbQ(), 1)
    dqMX = MX.sym("dqMX", biorbd_model.nbQ(), 1)
    aMX = MX.sym("aMX", biorbd_model.nbMuscles(), 1)
    uMX = MX.sym("uMX", biorbd_model.nbMuscles(), 1)
    force_func = Function(
        "MuscleForce",
        [qMX, dqMX, aMX, uMX],
        [muscles_forces(qMX, dqMX, aMX, uMX, biorbd_model)],
        ["qMX", "dqMX", "aMX", "uMX"],
        ["Force"],
    ).expand()
    # Build the graph
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T_mhe, x0=x_ref, nbGT=nbGT,
                      number_shooting_points=Ns_mhe, use_torque=use_torque, use_SX=use_ACADOS)
    if TRACK_EMG:
        if os.path.isfile(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt"):
            f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")
            f.write("Ns_mhe;  Co_lvl;  Marker_noise;  EMG_noise;  nb_try;  iter\n")
            f.close()
    else:
        if os.path.isfile(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt"):
            f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")
            f.write("Ns_mhe;  Co_lvl;  Marker_noise;  EMG_noise;  nb_try;  iter\n")
            f.close()
    # Loop for each co-contraction level
    for co in range(nb_co_lvl):
        # get initial guess
        motion = 'REACH2'
        with open(
                f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{co}.bob", 'rb'
        ) as file:
            data = pickle.load(file)
        states = data['data'][0]
        controls = data['data'][1]
        q_ref = states['q']
        dq_ref = states['q_dot']
        a_ref = states['muscles']
        u_ref = controls['muscles']
        w_tau = 'tau' in controls.keys()
        if w_tau:
            tau = controls['tau']
        else:
            tau = np.zeros((nbGT, Ns + 1))

        # if use_activation:
        #     x_ref_wt_noise = np.hstack([q_ref, dq_ref])
        # else:
        #     x_ref_wt_noise = np.hstack([q_ref, dq_ref, a_ref])

        # Loop for marker and EMG noise
        # if co == 0:
        #     marker_range = range(1, len(marker_noise_lvl))
        # else:
        marker_range = range(len(marker_noise_lvl))

        for marker_lvl in marker_range:
        # for marker_lvl in range(3, 4):
            if co == 0:
                emg_range = range(1, len(EMG_noise_lvl))
            else:
                emg_range = range(len(EMG_noise_lvl))
            for EMG_lvl in emg_range:
            # for EMG_lvl in range(2, 3):
                get_markers = markers_fun(biorbd_model)
                markers_target = np.zeros((3, biorbd_model.nbMarkers(), Ns + 1))
                for i in range(Ns + 1):
                    markers_target[:, :, i] = get_markers(q_ref[:, i])
                muscles_target = u_ref
                # Add electromechanical delay to the target
                if use_N_elec:
                    muscles_target = np.ndarray((u_ref.shape[0], u_ref.shape[1]))
                    for i in range(N_elec, u_ref.shape[1]):
                        muscles_target[:, i] = u_ref[:, i - N_elec]
                    for i in range(N_elec):
                        muscles_target[:, i] = muscles_target[:, N_elec]

                X_est_tries = np.ndarray((nb_try, X_est.shape[0], X_est.shape[1]))
                U_est_tries = np.ndarray((nb_try, U_est.shape[0], U_est.shape[1]))
                markers_target_tries = np.ndarray((nb_try, markers_target.shape[0], markers_target.shape[1], Ns+1))
                muscles_target_tries = np.ndarray((nb_try, muscles_target.shape[0], Ns+1))
                force_ref = np.ndarray((biorbd_model.nbMuscles(), Ns - Ns_mhe))
                force_est = np.ndarray((nb_try, biorbd_model.nbMuscles(), int(ceil(Ns / rt_ratio) - Ns_mhe)))
                if use_activation:
                    x_ref = np.concatenate((q_ref, dq_ref))
                else:
                    x_ref = np.concatenate((q_ref, dq_ref, a_ref))
                err_tries = np.ndarray((nb_try, 9))

                # Loop for simulate some tries, generate new random nosie to each iter

                for tries in range(nb_try):
                    print(f"--- Ns_mhe = {Ns_mhe}; Co_lvl: {co}; Marker_noise: {marker_lvl}; EMG_noise : {EMG_lvl}; nb_try : {tries} ---")

                    # Generate data with noise
                    if use_noise:
                        if marker_lvl != 0:
                            markers_target = generate_noise(
                                biorbd_model, q_ref, u_ref, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
                            )[0]

                        if EMG_lvl != 0:
                            muscles_target = generate_noise(
                                biorbd_model, q_ref, u_ref, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
                            )[1]

                    # reload the model with the real markers
                    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")

                    # set initial state
                    ocp.nlp[0].x_bounds.min[:, 0] = x_ref[:, 0]
                    ocp.nlp[0].x_bounds.max[:, 0] = x_ref[:, 0]

                    # set initial guess on state
                    x_init = InitialGuessOption(
                        x_ref[:, 0:Ns_mhe*rt_ratio+1:rt_ratio], interpolation=InterpolationType.EACH_FRAME)
                    u0 = muscles_target
                    u_init = InitialGuessOption(
                        u0[:, 0:Ns_mhe*rt_ratio:rt_ratio], interpolation=InterpolationType.EACH_FRAME)
                    ocp.update_initial_guess(x_init, u_init)

                    # Update objectives functions
                    w_marker = 10000000
                    w_control = 100000
                    objectives = ObjectiveList()
                    if TRACK_EMG:
                        w_marker = 100000000
                        w_torque = 10
                        objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=w_control,
                                       target=muscles_target[:, 0:Ns_mhe*rt_ratio:rt_ratio],
                                       )
                        if use_torque:
                            objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=w_torque)

                        objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)
                    else:
                        w_marker = 100000000
                        w_control = 100000
                        w_torque = 10
                        objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=w_control)
                        if use_torque:
                            objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=500)

                    objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=w_marker,
                                   target=markers_target[:, :, 0:(Ns_mhe+1)*rt_ratio:rt_ratio])
                    objectives.add(
                        Objective.Lagrange.MINIMIZE_STATE, weight=10, states_idx=np.array(range(biorbd_model.nbQ())))
                    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10,
                                   states_idx=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)))
                    if use_activation is not True:
                        objectives.add(
                            Objective.Lagrange.MINIMIZE_STATE,
                            weight=1,
                            states_idx=np.array(
                                range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles()))
                        )
                    ocp.update_objectives(objectives)

                    sol = ocp.solve(solver=Solver.ACADOS,
                                    show_online_optim=False,
                                    solver_options={
                                        "nlp_solver_tol_comp": 1e-4,
                                        "nlp_solver_tol_eq": 1e-4,
                                        "nlp_solver_tol_stat": 1e-4,
                                        "integrator_type": "IRK",
                                        "nlp_solver_type": "SQP",
                                        "sim_method_num_steps": 1,
                                        "print_level": 0,
                                        "nlp_solver_max_iter": 15,
                                    })
                    if sol['status'] != 0:
                        if TRACK_EMG:
                            f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")
                        else:
                            f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")

                        f.write(f"{Ns_mhe}; {co}; {marker_lvl}; {EMG_lvl}; {tries}; "
                                f"'init'\n")
                        f.close()
                    x0, u0, x_out, u_out = warm_start_mhe(ocp, sol, use_activation=use_activation)
                    X_est[:, 0] = x_out
                    U_est[:, 0] = u_out
                    tic = time()
                    for iter in range(1, ceil((Ns+1)/rt_ratio-Ns_mhe)):
                        # print(iter)
                        # set initial state
                        ocp.nlp[0].x_bounds.min[:, 0] = x0[:, 0]
                        ocp.nlp[0].x_bounds.max[:, 0] = x0[:, 0]

                        # set initial guess on state
                        # ocp.nlp[0].X_init.init = x0
                        # ocp.nlp[0].U_init.init = u0

                        x_init = InitialGuessOption(x0, interpolation=InterpolationType.EACH_FRAME)
                        u_init = InitialGuessOption(u0, interpolation=InterpolationType.EACH_FRAME)
                        ocp.update_initial_guess(x_init, u_init)

                        objectives = ObjectiveList()
                        if TRACK_EMG:
                            w_torque = 10
                            w_state = 10
                            objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=w_control,
                                           target=muscles_target[:, iter*rt_ratio:(Ns_mhe+iter)*rt_ratio:rt_ratio],
                                           )
                            objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)
                            if use_torque:
                                objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=w_torque)
                        else:
                            objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10000,
                                           )
                            if use_torque:
                                objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=500)

                        objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=w_marker,
                                       target=markers_target[:, :, iter*rt_ratio:(Ns_mhe+iter+1)*rt_ratio:rt_ratio])
                        objectives.add(
                            Objective.Lagrange.MINIMIZE_STATE, weight=1, states_idx=np.array(range(biorbd_model.nbQ())))
                        objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10,
                                       states_idx=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)))
                        if use_activation is not True:
                            objectives.add(
                                Objective.Lagrange.MINIMIZE_STATE,
                                weight=1,
                                states_idx=np.array(
                                    range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles()))
                            )
                        ocp.update_objectives(objectives)

                        sol = ocp.solve(solver=Solver.ACADOS,
                                        show_online_optim=False,
                                        solver_options={
                                            "nlp_solver_tol_comp": 1e-4,
                                            "nlp_solver_tol_eq": 1e-4,
                                            "nlp_solver_tol_stat": 1e-3,
                                            "integrator_type": "IRK",
                                            "nlp_solver_type": "SQP",
                                            "sim_method_num_steps": 1,
                                        })
                        x0, u0, x_out, u_out = warm_start_mhe(ocp, sol, use_activation=use_activation)
                        X_est[:, iter] = x_out
                        if iter < ceil(Ns/rt_ratio)-Ns_mhe:
                            U_est[:, iter] = u_out
                        if sol['status'] != 0:
                            if TRACK_EMG:
                                f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")
                            else:
                                f = open(f"{fold}status_track_rt_EMG{TRACK_EMG}.txt", "a")
                            f.write(f"{Ns_mhe}; {co}; {marker_lvl}; {EMG_lvl}; {tries}; "
                                    f"{iter}\n")
                            f.close()

                    q_est = X_est[:biorbd_model.nbQ(), :]
                    dq_est = X_est[biorbd_model.nbQ():biorbd_model.nbQ() * 2, :]
                    a_est = X_est[-biorbd_model.nbMuscles():, :]

                    for i in range(biorbd_model.nbMuscles()):
                        for j in range(int(ceil(Ns / rt_ratio) - Ns_mhe)):
                            force_est[tries, i, j] = force_func(
                                q_est[:, j], dq_est[:, j], a_est[:, j], U_est[:, j]
                            )[i, :]

                    toc = time() - tic
                    X_est_tries[tries, :, :] = X_est
                    U_est_tries[tries, :, :] = U_est
                    markers_target_tries[tries, :, :, :] = markers_target
                    muscles_target_tries[tries, :, :] = muscles_target


                    print(f"nb loops: {iter}")
                    print(f"Total time to solve with ACADOS : {toc} s")
                    print(f"Time per MHE iter. : {toc/iter} s")
                    err_offset = Ns_mhe
                    final_offset = 22
                    init_offset = 15
                    err = compute_err_mhe(init_offset, final_offset, err_offset, X_est, U_est, Ns, biorbd_model, q_ref,
                                      dq_ref, tau, a_ref, u_ref, nbGT, ratio=rt_ratio, use_activation=use_activation)

                    err_tmp = [Ns_mhe, rt_ratio, toc, toc/iter, err['q'], err['q_dot'], err['tau'], err['muscles'],
                               err['markers']]
                    err_tries[tries, :] = err_tmp
                    print(err)
                    # plt.subplot(211)
                    # for est, name in zip(X_est[:biorbd_model.nbQ(), :], biorbd_model.nameDof()):
                    #     plt.plot(est, 'x', label=name.to_string() + '_q_est')
                    # plt.gca().set_prop_cycle(None)
                    # for tru, name in zip(q_ref, biorbd_model.nameDof()):
                    #     plt.plot(tru, label=name.to_string() + '_q_tru')
                    # plt.legend()
                    #
                    # plt.subplot(212)
                    # for est, name in zip(X_est[biorbd_model.nbQ():biorbd_model.nbQ() * 2, :], biorbd_model.nameDof()):
                    #     plt.plot(est, 'x', label=name.to_string() + '_qdot_est')
                    # plt.gca().set_prop_cycle(None)
                    # for tru, name in zip(dq_ref, biorbd_model.nameDof()):
                    #     plt.plot(tru, label=name.to_string() + '_qdot_tru')
                    # plt.legend()
                    # plt.tight_layout()
                    # plt.figure('q')
                    # for i in range(biorbd_model.nbQ()):
                    #     plt.subplot(3, 2, i + 1)
                    #     plt.plot(X_est[i, :], 'x')
                    #     plt.plot(q_ref[i, 0:Ns+1:rt_ratio])
                    #     # plt.plot(muscles_target[i, :], 'k--')
                    #     plt.title(biorbd_model.muscleNames()[i].to_string())
                    # plt.legend(labels=['q_est', 'q_ref'], bbox_to_anchor=(1.05, 1), loc='upper left',
                    #            borderaxespad=0.)
                    # plt.figure('qdot')
                    # for i in range(biorbd_model.nbQ(), biorbd_model.nbQ()*2):
                    #     plt.subplot(3, 2, i-nbQ + 1)
                    #     plt.plot(X_est[i, :], 'x')
                    #     plt.plot(dq_ref[i-nbQ, 0:Ns + 1:rt_ratio])
                    #     # plt.plot(muscles_target[i, :], 'k--')
                    #     plt.title(biorbd_model.muscleNames()[i].to_string())
                    # plt.legend(labels=['q_est', 'q_ref'], bbox_to_anchor=(1.05, 1), loc='upper left',
                    #            borderaxespad=0.)
                    # plt.figure('Muscles excitations')
                    # for i in range(biorbd_model.nbMuscles()):
                    #     plt.subplot(4, 5, i + 1)
                    #     plt.plot(U_est[nbGT + i, :])
                    #     plt.plot(u_ref[i, 0:Ns:rt_ratio], c='red')
                    #     # plt.plot(muscles_target[i, :], 'k--')
                    #     plt.title(biorbd_model.muscleNames()[i].to_string())
                    # plt.legend(labels=['u_est', 'u_init', 'u_with_noise'], bbox_to_anchor=(1.05, 1), loc='upper left',
                    #            borderaxespad=0.)
                    # # plt.tight_layout()
                    #
                    # n_mark = biorbd_model.nbMarkers()
                    # get_markers = markers_fun(biorbd_model)
                    #
                    # markers = np.zeros((3, biorbd_model.nbMarkers(), q_ref.shape[1]))
                    # for i in range(q_ref.shape[1]):
                    #     markers[:, :, i] = get_markers(q_ref[:, i])
                    #
                    # markers_est = np.zeros((3, biorbd_model.nbMarkers(), X_est.shape[1]))
                    # for i in range(X_est.shape[1]):
                    #     markers_est[:, :, i] = get_markers(X_est[:biorbd_model.nbQ(), i])
                    #
                    # plt.figure("Markers")
                    # for i in range(markers_target.shape[1]):
                    #     plt.plot(markers_target[:, i, 0:Ns:rt_ratio].T, "k")
                    #     plt.plot(markers[:, i, 0:Ns:rt_ratio].T, "r--")
                    #     plt.plot(markers_est[:, i, :].T, "b")
                    # plt.xlabel("Time")
                    # plt.ylabel("Markers Position")
                    # plt.show()
                    print()

                for i in range(biorbd_model.nbMuscles()):
                    for k in range(int(Ns - Ns_mhe)):
                        force_ref[i, k] = force_func(q_ref[:, k], dq_ref[:, k], a_ref[:, k], u_ref[:, k])[i, :]
                # Write stats file for all tries
                # T_mhe_new = sum(T_mhe_tries)/nb_try
                # T_tot = sum(T_tot_tries) / nb_try
                err_dic = {"err_tries": err_tries}
                if WRITE_STATS:
                    if os.path.isfile(f"solutions/stats_rt_activation_driven{use_activation}.mat"):
                        matcontent = sio.loadmat(f"solutions/stats_rt_activation_driven{use_activation}.mat")
                        err_mat = np.concatenate((matcontent['err_tries'], err_tries))
                        err_dic = {"err_tries": err_mat}
                        sio.savemat(f"solutions/stats_rt_activation_driven{use_activation}.mat", err_dic)
                    else:
                        sio.savemat(f"solutions/stats_rt_activation_driven{use_activation}.mat", err_dic)

                # Save results for all tries
                if save_results:
                    dic = {
                        "X_est": X_est_tries,
                        "U_est": U_est_tries,
                        "x_sol": x_ref,
                        "u_sol": u_ref,
                        "markers_target": markers_target_tries,
                        "u_target": muscles_target_tries,
                        "time_per_mhe": toc/(iter),
                        "time_tot": toc,
                        "co_lvl": co,
                        "marker_noise_lvl": marker_noise_lvl[marker_lvl],
                        "EMG_noise_lvl": EMG_noise_lvl[EMG_lvl],
                        "N_mhe": Ns_mhe,
                        "N_tot": Ns,
                        "rt_ratio": rt_ratio,
                        "f_est": force_est,
                        "f_ref": force_ref}
                    if TRACK_EMG:
                        sio.savemat(
                            f"{fold}track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat",
                            dic
                        )
                    else:
                        sio.savemat(
                            f"{fold}track_mhe_wt_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat",
                            dic
                        )

# plt.subplot(211)
# for est, name in zip(X_est[:biorbd_model.nbQ(), :], biorbd_model.nameDof()):
#     plt.plot(est, 'x', label=name.to_string() + '_q_est')
# plt.gca().set_prop_cycle(None)
# for tru, name in zip(q_ref, biorbd_model.nameDof()):
#     plt.plot(tru, label=name.to_string() + '_q_tru')
# plt.legend()
#
# plt.subplot(212)
# for est, name in zip(X_est[biorbd_model.nbQ():biorbd_model.nbQ() * 2, :], biorbd_model.nameDof()):
#     plt.plot(est, 'x', label=name.to_string() + '_qdot_est')
# plt.gca().set_prop_cycle(None)
# for tru, name in zip(dq_ref, biorbd_model.nameDof()):
#     plt.plot(tru, label=name.to_string() + '_qdot_tru')
# plt.legend()
# plt.tight_layout()
#
# if use_torque:
#     plt.figure()
#     plt.subplot(211)
#     for est, name in zip(U_est[:nbGT, :], biorbd_model.nameDof()):
#         plt.plot(est, 'x', label=name.to_string() + '_tau_est')
#     plt.gca().set_prop_cycle(None)
#     for tru, name in zip(tau, biorbd_model.nameDof()):
#         plt.plot(tru, label=name.to_string() + '_tau_tru')
#     plt.legend()
#     plt.subplot(212)
#
# plt.figure('Muscles excitations')
# for i in range(biorbd_model.nbMuscles()):
#     plt.subplot(4, 5, i + 1)
#     plt.plot(U_est[nbGT + i, :])
#     plt.plot(u_ref[i, :], c='red')
#     plt.plot(muscles_target[i, :], 'k--')
#     plt.title(biorbd_model.muscleNames()[i].to_string())
# plt.legend(labels=['u_est', 'u_init', 'u_with_noise'], bbox_to_anchor=(1.05, 1), loc='upper left',
#            borderaxespad=0.)
# # plt.tight_layout()
#
# n_mark = biorbd_model.nbMarkers()
# get_markers = markers_fun(biorbd_model)
# markers = np.zeros((3, biorbd_model.nbMarkers(), q_ref.shape[1]))
# for i in range(q_ref.shape[1]):
#     markers[:, :, i] = get_markers(q_ref[:, i])
# markers_est = np.zeros((3, biorbd_model.nbMarkers(), X_est.shape[1]))
# for i in range(X_est.shape[1]):
#     markers_est[:, :, i] = get_markers(X_est[:biorbd_model.nbQ(), i])
# plt.figure("Markers")
# for i in range(markers_target.shape[1]):
#     plt.plot(markers_target[:, i, :].T, "k")
#     plt.plot(markers[:, i, :].T, "r--")
#     plt.plot(markers_est[:, i, :].T, "b")
# plt.xlabel("Time")
# plt.ylabel("Markers Position")
# # plt.show()
# print()
