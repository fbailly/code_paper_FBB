import biorbd
from time import time
import numpy as np
from casadi import MX, Function
import matplotlib.pyplot as plt
import pickle
import sys
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
    InitialGuessList,
    InitialGuessOption,
    ShowResult,
    Solver,
    InterpolationType,
    Bounds,
)
import os
import scipy.io as sio

def prepare_ocp(
        biorbd_model,
        final_time,
        x0,
        nbGT,
        number_shooting_points,
        use_SX=False,
        nb_threads=1,
        use_activation=True,
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
    use_activation = False
    use_torque = False
    use_ACADOS = True
    use_bash = True
    save_stats = True
    if use_activation:
        use_N_elec = True
    else:
        use_N_elec = False

    N_elec = 2
    T_elec = 0.02
    T = 8
    Ns = 800
    final_offset = 22
    init_offset = 15
    # if use_N_elec:
    #     Ns = Ns - N_elec

    motion = 'REACH2'
    i = '0'
    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
    with open(
            f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{i}.bob", 'rb'
    ) as file:
        data = pickle.load(file)
    states = data['data'][0]
    controls = data['data'][1]
    q_ref = states['q']
    dq_ref = states['q_dot']
    a_ref = states['muscles']
    u_ref = controls['muscles']
    if use_torque:
        nbGT = biorbd_model.nbGeneralizedTorque()
    else:
        nbGT = 0
    nbMT = biorbd_model.nbMuscleTotal()
    nbQ = biorbd_model.nbQ()
    w_tau = 'tau' in controls.keys()
    if w_tau:
        tau = controls['tau']
    else:
        tau = np.zeros((nbGT, Ns+1))
    if use_activation:
        x0 = np.hstack([q_ref[:, 0], dq_ref[:, 0]])
    else:
        x0 = np.hstack([q_ref[:, 0], dq_ref[:, 0], a_ref[:, 0]])
    tau_init = 0
    muscle_init = 0.5

    # get targets
    get_markers = markers_fun(biorbd_model)
    markers_target = np.zeros((3, biorbd_model.nbMarkers(), Ns+1))
    for i in range(Ns+1):
        markers_target[:, :, i] = get_markers(q_ref[:, i])
    muscles_target = u_ref

    muscles_target_real = np.ndarray((u_ref.shape[0], u_ref.shape[1]))
    for i in range(N_elec, u_ref.shape[1]):
        muscles_target_real[:, i] = u_ref[:, i-N_elec]
    for i in range(N_elec):
        muscles_target_real[:, i] = muscles_target_real[:, N_elec]

    # plt.figure()
    # plt.plot(muscles_target_real[:, :].T, 'x')
    # plt.gca().set_prop_cycle(None)
    # plt.plot(u_ref[:, :].T)
    # plt.plot(a_ref[:, :].T, 'o')
    # plt.legend(
    #     labels=['Muscle excitation estimate', 'Muscle excitation truth'],
    #     bbox_to_anchor=(1, 1),
    #     loc='upper left', borderaxespad=0.
    # )
    # plt.tight_layout()
    # plt.show()
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T, x0=x0, nbGT=nbGT,
                      number_shooting_points=Ns, use_torque=use_torque, use_activation=use_activation, use_SX=use_ACADOS)

    # set initial state
    ocp.nlp[0].x_bounds.min[:, 0] = x0
    ocp.nlp[0].x_bounds.max[:, 0] = x0

    # set initial guess on state
    x_init = InitialGuessOption(x0, interpolation=InterpolationType.CONSTANT)
    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)
    u_init = InitialGuessOption(u0, interpolation=InterpolationType.CONSTANT)
    ocp.update_initial_guess(x_init, u_init)

    objectives = ObjectiveList()
    if use_activation:
        objectives.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=10000, target=muscles_target_real[:, :-1])
    else:
        objectives.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=1000, target=muscles_target[:, :-1])

    objectives.add(Objective.Lagrange.TRACK_MARKERS, weight=100000000, target=markers_target[:, :, :])
    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=1, states_idx=np.array(range(nbQ)))
    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10, states_idx=np.array(range(nbQ, nbQ * 2)))
    if use_activation is not True:
        objectives.add(
            Objective.Lagrange.MINIMIZE_STATE, weight=1, states_idx=np.array(range(nbQ * 2, nbQ * 2 + nbMT))
        )
    if use_torque:
        objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=10)
    ocp.update_objectives(objectives)

    if use_ACADOS:
        tic = time()
        sol = ocp.solve(solver=Solver.ACADOS,
                        show_online_optim=False,
                        solver_options={
                            "nlp_solver_tol_comp": 1e-6,
                            "nlp_solver_tol_eq": 1e-6,
                            "nlp_solver_tol_stat": 1e-6,
                            "integrator_type": "IRK",
                            "nlp_solver_type": "SQP",
                            "sim_method_num_steps": 1,
                        })
        print(f"Time to solve with ACADOS : {time()-tic} s")
    else:
        tic = time()
        sol = ocp.solve(
            solver=Solver.IPOPT,
            show_online_optim=False,
            solver_options={
                "tol": 1e-4,
                "dual_inf_tol": 1e-6,
                "constr_viol_tol": 1e-6,
                "compl_inf_tol": 1e-6,
                "linear_solver": "ma57",
                "max_iter": 500,
                "hessian_approximation": "exact",
            })
        print(f"Time to solve with IPOPT : {time() - tic} s")

    toc = time() - tic
    print(f"Total time to solve with ACADOS : {toc} s")

    data_est = Data.get_data(ocp, sol)
    if use_activation:
        X_est = np.vstack([data_est[0]['q'], data_est[0]['q_dot']])
    else:
        X_est = np.vstack([data_est[0]['q'], data_est[0]['q_dot'], data_est[0]['muscles']])
    if use_torque:
        U_est = np.vstack([data_est[1]['tau'], data_est[1]['muscles']])
    else:
        U_est = data_est[1]['muscles']

    err = compute_err(init_offset, final_offset, X_est, U_est, Ns, biorbd_model, q_ref,
                      dq_ref, tau, a_ref, u_ref, nbGT, use_activation=use_activation)

    use_noise = False
    print(err)
    err_tmp = np.array([[Ns, 1, toc, toc, err['q'], err['q_dot'], err['tau'], err['muscles'], err['markers']]])
    if save_stats:
        if os.path.isfile(f"solutions/stats_rt_activation_driven{use_activation}.mat"):
            matcontent = sio.loadmat(
                f"solutions/stats_rt_activation_driven{use_activation}.mat")
            err_mat = np.concatenate((matcontent['err_tries'], err_tmp))
            err_dic = {"err_tries": err_mat}
            sio.savemat(f"solutions/stats_rt_activation_driven{use_activation}.mat", err_dic)
        else:
            RuntimeError(f"File 'solutions/stats_rt_activation_driven{use_activation}.mat' does not exist")
    # plt.subplot(211)
    # plt.plot(X_est[:biorbd_model.nbQ(), :].T, 'x')
    # plt.gca().set_prop_cycle(None)
    # plt.plot(q_ref.T)
    # plt.legend(labels=['Q estimate', 'Q truth'], bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    # plt.subplot(212)
    # plt.plot(X_est[biorbd_model.nbQ():biorbd_model.nbQ()*2, :].T, 'x')
    # plt.gca().set_prop_cycle(None)
    # plt.plot(dq_ref.T)
    # plt.legend(labels=['Qdot estimate', 'Qdot truth'], bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    # # plt.tight_layout()
    # plt.figure('Muscles excitations')
    # for i in range(biorbd_model.nbMuscles()):
    #     plt.subplot(4, 5, i + 1)
    #     # if use_N_elec:
    #     #     plt.plot(a_ref[i, :-N_elec].T)
    #     #     plt.plot(u_ref[i, :-N_elec].T, '--')
    #     # else:
    #     plt.plot(a_ref[i, :].T)
    #     plt.plot(u_ref[i, :].T, '--')
    #     plt.plot(U_est[i, :].T, 'k:')
    #     plt.title(biorbd_model.muscleNames()[i].to_string())
    # plt.legend(
    #     labels=['a_ref', 'u_ref', 'a_est'], bbox_to_anchor=(1.05, 1), loc='upper left',
    #     borderaxespad=0.
    # )
    # plt.figure('RMSE_activations')
    # plt.figure()
    # if use_torque:
    #     plt.subplot(211)
    #     plt.plot(U_est[:nbGT, :].T, 'x', label='Tau estimate')
    #     plt.gca().set_prop_cycle(None)
    #     plt.plot(tau.T)
    #     plt.legend(labels=['Tau estimate', 'Tau truth'], bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    #     plt.subplot(212)
    # plt.plot(U_est[nbGT:, :].T, 'x')
    # plt.gca().set_prop_cycle(None)
    # if use_activation:
    #     plt.plot(a_ref[:, N_elec:].T)
    #     plt.gca().set_prop_cycle(None)
    #     plt.plot(u_ref[:, N_elec:].T, '--')
    # else:
    #     plt.plot(u_ref.T)
    # plt.legend(
    #     labels=['Muscle excitation estimate', 'Muscle excitation truth'],
    #     bbox_to_anchor=(1, 1),
    #     loc='upper left', borderaxespad=0.
    # )
    # plt.tight_layout()
    # plt.show()
    # if use_activation:
    #     ocp.save_get_data(
    #         sol, f"solutions/tracking_markers_EMG_activations_driven.bob"
    #     )
    # else:
    #     ocp.save_get_data(
    #         sol, f"solutions/tracking_markers_EMG_excitations_driven.bob"
    #     )
    # --- Show results --- #
    # result = ShowResult(ocp, sol)
    # result.graphs()
    # result.animate()
