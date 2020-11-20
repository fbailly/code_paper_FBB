import biorbd
from time import time
import numpy as np
from casadi import MX, Function
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
import sys
from generate_data_noise_funct import generate_noise
import os
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    Data,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    InitialGuessOption,
    ShowResult,
    Solver,
    InterpolationType,
    Bounds,
)

def compute_err(Ns_mhe, X_est, U_est, Ns, model, q, dq, tau, activations, excitations, nbGT):
    model = model
    get_markers = markers_fun(model)
    err = dict()
    nbGT = nbGT
    Ns = Ns
    norm_err = np.sqrt(Ns-Ns_mhe)
    q_ref = q[:, :-Ns_mhe]
    dq_ref = dq[:, :-Ns_mhe]
    tau_ref = tau[:, :-Ns_mhe-1]
    muscles_ref = excitations[:, :-Ns_mhe - 1]
    if use_activation:
        muscles_ref = activations[:, :-Ns_mhe - 1]
    sol_mark = np.zeros((3, model.nbMarkers(), Ns+1-Ns_mhe))
    sol_mark_ref = np.zeros((3, model.nbMarkers(), Ns+1-Ns_mhe))
    err['q'] = np.linalg.norm(X_est[:model.nbQ(), :]-q_ref)/norm_err
    err['q_dot'] = np.linalg.norm(X_est[model.nbQ():model.nbQ()*2, :]-dq_ref)/norm_err
    err['tau'] = np.linalg.norm(U_est[:nbGT, :]-tau_ref)/norm_err
    err['muscles'] = np.linalg.norm(U_est[nbGT:, :]-muscles_ref)/norm_err
    for i in range(Ns+1-Ns_mhe):
        sol_mark[:, :, i] = get_markers(X_est[:model.nbQ(), i])
        sol_mark_ref[:, :, i] = get_markers(q[:, i])
    err['markers'] = np.linalg.norm(sol_mark - sol_mark_ref)/norm_err
    return err


def warm_start_mhe(ocp, sol):
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


def markers_fun(biorbd_model):
    qMX = MX.sym('qMX', biorbd_model.nbQ())
    return Function('markers', [qMX], [biorbd_model.markers(qMX)])


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
    save_results = False
    TRACK_EMG = True
    use_noise = False
    use_co = False
    use_bash = False
    use_try = False

    # Variable of the problem
    Ns = 400
    Ns_mhe = 2
    if use_bash:
        Ns_mhe = int(sys.argv[1])
    T = 4
    T_mhe = T / Ns * Ns_mhe
    if use_try:
        nb_try = 10
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
        x0_ref = np.zeros((biorbd_model.nbQ()*2))
    else:
        x0_ref = np.zeros((biorbd_model.nbQ()*2 + biorbd_model.nbMuscles()))

    # Define noise
    if use_noise:
        marker_noise_lvl = [0, 0.002, 0.005, 0.01]
        EMG_noise_lvl = [0, 0.05, 0.1, 0.2]
    else:
        marker_noise_lvl = [0]
        EMG_noise_lvl = [0]

    # define x_est u_est size
    if use_activation:
        X_est = np.zeros((biorbd_model.nbQ() * 2, Ns + 1 - Ns_mhe))
    else:
        X_est = np.zeros((biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), Ns + 1 - Ns_mhe))
    U_est = np.zeros((nbGT + biorbd_model.nbMuscleTotal(), Ns - Ns_mhe))

    # Set number of co-contraction level
    nb_co_lvl = 1
    if use_co:
        nb_co_lvl = 4
    # Build the graph
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T_mhe, x0=x0_ref, nbGT=nbGT,
                      number_shooting_points=Ns_mhe, use_torque=use_torque, use_SX=use_ACADOS)
    if TRACK_EMG:
        # if os.path.isfile(f"solutions/w_track_low_weight_3/status_track_EMG{TRACK_EMG}.txt"):
        #     os.remove(f"solutions/w_track_low_weight_3/status_track_EMG{TRACK_EMG}.txt")
        f = open(f"solutions/w_track_low_weight_3/status_track_EMG{TRACK_EMG}.txt", "a")
    else:
        # if os.path.isfile(f"solutions/wt_track_low_weight_3/status_track_EMG{TRACK_EMG}.txt"):
        #     os.remove(f"solutions/wt_track_low_weight_3/status_track_EMG{TRACK_EMG}.txt")
        f = open(f"solutions/wt_track_low_weight_3/status_track_EMG{TRACK_EMG}.txt", "a")
    # f.write("Ns_mhe;  Co_lvl;  Marker_noise;  EMG_noise;  nb_try;  iter\n")
    f.close()
    # Loop for each co-contraction level
    if TRACK_EMG:
        folder = ["solutions/w_track_low_weight_3", "solutions/w_track_low_weight_2", "solutions/with_track_emg"]
    else:
        folder = ["solutions/wt_track_low_weight_3", "solutions/wt_track_low_weight_2", "solutions/wt_track_emg"]
    fold = "solutions/wt_track_emg"
    w_marker = 100000000
    w_state = 10

    # fold = "solutions/wt_track_low_weight_2"
    # w_marker = 500000
    # w_state = 10

    # fold = "solutions/wt_track_low_weight_3"
    # w_marker = 500000
    # w_state = 100
    for co in range(nb_co_lvl):
        # get initial guess
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

        if use_activation:
            x0_ref = np.hstack([q_sol[:, 0], dq_sol[:, 0]])
        else:
            x0_ref = np.hstack([q_sol[:, 0], dq_sol[:, 0], a_sol[:, 0]])

        # Loop for marker and EMG noise
        for marker_lvl in range(len(marker_noise_lvl)):
        # for marker_lvl in range(3, 4):
            if TRACK_EMG:
                a = len(EMG_noise_lvl)
            else:
                a = 1
        #     for EMG_lvl in range(len(EMG_noise_lvl)):
            for EMG_lvl in range(a):
                get_markers = markers_fun(biorbd_model)
                markers_target = np.zeros((3, biorbd_model.nbMarkers(), Ns + 1))
                for i in range(Ns + 1):
                    markers_target[:, :, i] = get_markers(q_sol[:, i])
                muscles_target = u_sol

                X_est_tries = np.ndarray((nb_try, X_est.shape[0], Ns - Ns_mhe + 1))
                U_est_tries = np.ndarray((nb_try, U_est.shape[0], Ns - Ns_mhe))
                markers_target_tries = np.ndarray((nb_try, markers_target.shape[0], markers_target.shape[1], Ns + 1))
                muscles_target_tries = np.ndarray((nb_try, muscles_target.shape[0], Ns + 1))
                if use_activation:
                    x_sol = np.concatenate((q_sol, dq_sol))
                else:
                    x_sol = np.concatenate((q_sol, dq_sol, a_sol))
                err_tries = np.ndarray((nb_try, 7))

                # Loop for simulate some tries, generate new random nosie to each iter
                for tries in range(nb_try):
                    print(f"--- Ns_mhe = {Ns_mhe}; Co_lvl: {co}; Marker_noise: {marker_lvl}; EMG_noise : {EMG_lvl}; nb_try : {tries} ---")

                    # Generate data with noise
                    if use_noise:
                        if marker_lvl != 0:
                            markers_target = generate_noise(
                                biorbd_model, q_sol, u_sol, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
                            )[0]

                        if EMG_lvl != 0:
                            muscles_target = generate_noise(
                                biorbd_model, q_sol, u_sol, marker_noise_lvl[marker_lvl], EMG_noise_lvl[EMG_lvl]
                            )[1]

                    # reload the model with the real markers
                    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")

                    # set initial state
                    ocp.nlp[0].x_bounds.min[:, 0] = x0_ref
                    ocp.nlp[0].x_bounds.max[:, 0] = x0_ref

                    # set initial guess on state
                    x_init = InitialGuessOption(x0_ref, interpolation=InterpolationType.CONSTANT)
                    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)
                    u_init = InitialGuessOption(u0, interpolation=InterpolationType.CONSTANT)
                    ocp.update_initial_guess(x_init, u_init)

                    # Update objectives functions
                    # w_marker = 500000
                    # w_state = 100
                    objectives = ObjectiveList()
                    if TRACK_EMG:
                        w_control = 100000
                        w_torque = 10
                        objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=w_control,
                                       target=muscles_target[:, :Ns_mhe],
                                       )
                        if use_torque:
                            objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=w_torque)

                    else:
                        w_control = 10000
                        w_torque = 10
                        objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=w_control)
                        if use_torque:
                            objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=500)

                    objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=w_marker,
                                   target=markers_target[:, :, :Ns_mhe+1])
                    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=w_state)
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
                                    })
                    if sol['status'] != 0:
                        if TRACK_EMG:
                            f = open(f"{fold}/status_track_EMG{TRACK_EMG}.txt", "a")
                        else:
                            f = open(f"{fold}/status_track_EMG{TRACK_EMG}.txt", "a")

                        f.write(f"{Ns_mhe}; {co}; {marker_lvl}; {EMG_lvl}; {tries}; "
                                f"'init'\n")
                        f.close()
                    x0, u0, x_out, u_out = warm_start_mhe(ocp, sol)
                    X_est[:, 0] = x_out
                    U_est[:, 0] = u_out
                    tic = time()
                    cnt = 0
                    for iter in range(1, Ns-Ns_mhe+1):
                        cnt += 1
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
                            w_control = 100000
                            w_torque = 10
                            objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=w_control,
                                           target=muscles_target[:, iter:Ns_mhe+iter],
                                           )
                            if use_torque:
                                objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=w_torque)
                        else:
                            objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10000,
                                           )
                            if use_torque:
                                objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=500)

                        objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=w_marker,
                                       target=markers_target[:, :, iter:Ns_mhe+iter+1])
                        objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=w_state)
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
                                        })
                        x0, u0, x_out, u_out = warm_start_mhe(ocp, sol)
                        X_est[:, iter] = x_out
                        if iter < Ns-Ns_mhe:
                            U_est[:, iter] = u_out
                        if sol['status'] != 0:
                            if TRACK_EMG:
                                f = open(f"solutions/w_track_low_weight_3/status_track_EMG{TRACK_EMG}.txt", "a")
                            else:
                                f = open(f"solutions/wt_track_low_weight_3/status_track_EMG{TRACK_EMG}.txt", "a")
                            f.write(f"{Ns_mhe}; {co}; {marker_lvl}; {EMG_lvl}; {tries}; "
                                    f"{iter}\n")
                            f.close()
                    plt.subplot(211)
                    for est, name in zip(X_est[:biorbd_model.nbQ(), :], biorbd_model.nameDof()):
                        plt.plot(est, 'x', label=name.to_string() + '_q_est')
                    plt.gca().set_prop_cycle(None)
                    for tru, name in zip(q_sol, biorbd_model.nameDof()):
                        plt.plot(tru, label=name.to_string() + '_q_tru')
                    plt.legend()

                    plt.subplot(212)
                    for est, name in zip(X_est[biorbd_model.nbQ():biorbd_model.nbQ() * 2, :], biorbd_model.nameDof()):
                        plt.plot(est, 'x', label=name.to_string() + '_qdot_est')
                    plt.gca().set_prop_cycle(None)
                    for tru, name in zip(dq_sol, biorbd_model.nameDof()):
                        plt.plot(tru, label=name.to_string() + '_qdot_tru')
                    plt.legend()
                    plt.tight_layout()

                    plt.figure('Muscles excitations')
                    for i in range(biorbd_model.nbMuscles()):
                        plt.subplot(4, 5, i + 1)
                        plt.plot(U_est[nbGT + i, :])
                        plt.plot(u_sol[i, :], c='red')
                        plt.plot(muscles_target[i, :], 'k--')
                        plt.title(biorbd_model.muscleNames()[i].to_string())
                    plt.legend(labels=['u_est', 'u_init', 'u_with_noise'], bbox_to_anchor=(1.05, 1), loc='upper left',
                               borderaxespad=0.)
                    # plt.tight_layout()

                    n_mark = biorbd_model.nbMarkers()
                    get_markers = markers_fun(biorbd_model)
                    markers = np.zeros((3, biorbd_model.nbMarkers(), q_sol.shape[1]))
                    for i in range(q_sol.shape[1]):
                        markers[:, :, i] = get_markers(q_sol[:, i])
                    markers_est = np.zeros((3, biorbd_model.nbMarkers(), X_est.shape[1]))
                    for i in range(X_est.shape[1]):
                        markers_est[:, :, i] = get_markers(X_est[:biorbd_model.nbQ(), i])
                    plt.figure("Markers")
                    for i in range(markers_target.shape[1]):
                        plt.plot(markers_target[:, i, :].T, "k")
                        plt.plot(markers[:, i, :].T, "r--")
                        plt.plot(markers_est[:, i, :].T, "b")
                    plt.xlabel("Time")
                    plt.ylabel("Markers Position")
                    plt.show()
                    print()
                    X_est_tries[tries, :, :] = X_est
                    U_est_tries[tries, :, :] = U_est
                    markers_target_tries[tries, :, :, :] = markers_target
                    muscles_target_tries[tries, :, :] = muscles_target

                    toc = time() - tic
                    print(f"nb loops: {cnt}")
                    print(f"Total time to solve with ACADOS : {toc} s")
                    print(f"Time per MHE iter. : {toc/(Ns-Ns_mhe)} s")

                    err_offset = Ns_mhe + 1
                    err = compute_err(
                        err_offset,
                        X_est[:, :-err_offset+Ns_mhe],
                        U_est[:, :-err_offset+Ns_mhe], Ns, biorbd_model, q_sol, dq_sol, tau, a_sol, u_sol, nbGT)

                    err_tmp = [Ns_mhe, toc / (Ns - Ns_mhe), err['q'], err['q_dot'], err['tau'], err['muscles'],
                               err['markers']]
                    err_tries[tries, :] = err_tmp
                    print(err)

                # Write stats file for all tries

                err_dic = {"err_tries": err_tries}
                if WRITE_STATS:
                    if os.path.isfile(f"solutions/stats_ac_activation_driven{use_activation}.mat"):
                        matcontent = sio.loadmat(f"solutions/stats_ac_activation_driven{use_activation}.mat")
                        err_mat = np.concatenate((matcontent['err_tries'], err_tries))
                        err_dic = {"err_tries": err_mat}
                        sio.savemat(f"solutions/stats_ac_activation_driven{use_activation}.mat", err_dic)
                    else:
                        sio.savemat(f"solutions/stats_ac_activation_driven{use_activation}.mat", err_dic)

                # Save results for all tries
                if save_results:
                    # dic = {
                    #     "X_est": X_est_tries,
                    #     "U_est": U_est_tries,
                    #     "x_sol": x_sol,
                    #     "u_sol": u_sol,
                    #     "markers_target": markers_target_tries,
                    #     "u_target": muscles_target_tries,
                    #     "time_per_mhe": toc / (Ns - Ns_mhe),
                    #     "time_tot": toc,
                    #     "co_lvl": co,
                    #     "marker_noise_lvl": marker_noise_lvl[marker_lvl],
                    #     "EMG_noise_lvl": EMG_noise_lvl[EMG_lvl],
                    #     "N_mhe": Ns_mhe,
                    #     "N_tot": Ns}
                    if TRACK_EMG:
                        mat_content = sio.loadmat(
                            f"{fold}/track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                        )
                        X_est_augmented = np.concatenate((mat_content['X_est'], X_est_tries))
                        U_est_augmented = np.concatenate((mat_content['U_est'], U_est_tries))
                        marker_target_augmented = np.concatenate((mat_content["markers_target"], markers_target_tries))
                        muscles_target_augmented = np.concatenate((mat_content["u_target"], muscles_target_tries))
                        mat_content["X_est"] = X_est_augmented
                        mat_content["U_est"] = U_est_augmented
                        mat_content["markers_target"] = marker_target_augmented
                        mat_content["u_target"] = muscles_target_augmented
                        sio.savemat(
                            f"{fold}/track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat",
                        mat_content)
                    else:
                        mat_content = sio.loadmat(
                            f"{fold}/track_mhe_wt_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                        )
                        X_est_augmented = np.concatenate((mat_content['X_est'], X_est_tries))
                        U_est_augmented = np.concatenate((mat_content['U_est'], U_est_tries))
                        marker_target_augmented = np.concatenate(
                            (mat_content["markers_target"], markers_target_tries))
                        muscles_target_augmented = np.concatenate((mat_content["u_target"], muscles_target_tries))
                        mat_content["X_est"] = X_est_augmented
                        mat_content["U_est"] = U_est_augmented
                        mat_content["markers_target"] = marker_target_augmented
                        mat_content["u_target"] = muscles_target_augmented
                        sio.savemat(
                            f"{fold}/track_mhe_wt_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat",
                            mat_content)


# plt.subplot(211)
# for est, name in zip(X_est[:biorbd_model.nbQ(), :], biorbd_model.nameDof()):
#     plt.plot(est, 'x', label=name.to_string() + '_q_est')
# plt.gca().set_prop_cycle(None)
# for tru, name in zip(q_sol, biorbd_model.nameDof()):
#     plt.plot(tru, label=name.to_string() + '_q_tru')
# plt.legend()
#
# plt.subplot(212)
# for est, name in zip(X_est[biorbd_model.nbQ():biorbd_model.nbQ() * 2, :], biorbd_model.nameDof()):
#     plt.plot(est, 'x', label=name.to_string() + '_qdot_est')
# plt.gca().set_prop_cycle(None)
# for tru, name in zip(dq_sol, biorbd_model.nameDof()):
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
#     plt.plot(u_sol[i, :], c='red')
#     plt.plot(muscles_target[i, :], 'k--')
#     plt.title(biorbd_model.muscleNames()[i].to_string())
# plt.legend(labels=['u_est', 'u_init', 'u_with_noise'], bbox_to_anchor=(1.05, 1), loc='upper left',
#            borderaxespad=0.)
# # plt.tight_layout()
#
# n_mark = biorbd_model.nbMarkers()
# get_markers = markers_fun(biorbd_model)
# markers = np.zeros((3, biorbd_model.nbMarkers(), q_sol.shape[1]))
# for i in range(q_sol.shape[1]):
#     markers[:, :, i] = get_markers(q_sol[:, i])
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
