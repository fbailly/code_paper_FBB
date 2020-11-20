import biorbd
from time import time
import numpy as np
from casadi import MX, Function
import bioviz
import pickle
import os.path
import matplotlib.pyplot as plt
# from generate_data_noise_funct import generate_noise
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
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
    BoundsOption,
    Instant,
    Data
)

def markers_fun(biorbd_model):
    qMX = MX.sym('qMX', biorbd_model.nbQ())
    return Function('markers', [qMX], [biorbd_model.markers(qMX)])

def prepare_ocp(
        biorbd_model,
        final_time,
        number_shooting_points,
        x0,
        xT,
        use_SX=False,
        nb_threads=8,
        ):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd_model
    nbQ = biorbd_model.nbQ()

    # tau_min, tau_max, tau_init = -10, 10, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1
    excitation_min, excitation_max, excitation_init = 0, 1, 0.2

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_STATE, weight=10, states_idx=np.array(range(biorbd_model.nbQ())))
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=10,
                            states_idx=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)))
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_STATE,
        weight=10,
        states_idx=np.array(range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles()))
    )
    objective_functions.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)

    objective_functions.add(
        Objective.Mayer.TRACK_STATE,
        weight=100000,
        target=np.array([xT[:biorbd_model.nbQ()]]).T,
        states_idx=np.array(range(biorbd_model.nbQ()))
    )
    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    # add muscle activation bounds
    x_bounds[0].concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )
    x_bounds[0].min[:nbQ, 0] = [-0.1, -0.3, 0.1, -0.3]
    x_bounds[0].max[:nbQ, 0] = [-0.1, 0, 0.3, 0]
    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [excitation_min] * biorbd_model.nbMuscleTotal(),
            [excitation_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    # Initial guesses
    x_init = InitialGuessOption(np.tile(np.concatenate(
        (x0, [activation_init] * biorbd_model.nbMuscles()))
        , (number_shooting_points+1, 1)).T, interpolation=InterpolationType.EACH_FRAME)
    u0 = np.array([excitation_init]*biorbd_model.nbMuscles())
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
    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
    T = 0.8
    Ns = 100
    co_value = []
    motion = 'REACH2'
    x0 = np.array([0., -0.2, 0, 0, 0, 0, 0, 0])
    xT = np.array([-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0])

    use_ACADOS = True
    use_IPOPT = False
    use_BO = False
    use_CO = True
    save_data = False

    excitations_max = [1] * biorbd_model.nbMuscleTotal()
    if use_CO is not True:
        excitations_init = [[0.05] * biorbd_model.nbMuscleTotal()]
        excitations_min = [[0] * biorbd_model.nbMuscleTotal()]

    else:
        excitations_init = [
            [0.05] * biorbd_model.nbMuscleTotal(),
            [0.05] * 6 + [0.1] * 3 + [0.05] * 10,
            [0.05] * 6 + [0.2] * 3 + [0.05] * 10,
            [0.05] * 6 + [0.3] * 3 + [0.05] * 10,
            # [0.05] * 6 + [0.4] * 3 + [0.05] * 10
        ]
        excitations_min = [
            [0] * biorbd_model.nbMuscleTotal(),
            [0] * 6 + [0.1] * 3 + [0] * 10,
            [0] * 6 + [0.2] * 3 + [0] * 10,
            [0] * 6 + [0.3] * 3 + [0] * 10,
            # [0] * 6 + [0.4] * 3 + [0] * 10
        ]

    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T, number_shooting_points=Ns, x0=x0, xT=xT, use_SX=True)

    for i in range(0, len(excitations_init)):
        u_i = excitations_init[i]
        u_mi = excitations_min[i]
        u_ma = excitations_max

        # Update u_init and u_bounds
        u_init = InitialGuessOption(np.tile(u_i, (ocp.nlp[0].ns, 1)).T, interpolation=InterpolationType.EACH_FRAME)
        x_init = InitialGuessOption(np.tile(np.concatenate(
            (x0, [0.1] * biorbd_model.nbMuscles())), (ocp.nlp[0].ns + 1, 1)).T, interpolation=InterpolationType.EACH_FRAME)
        ocp.update_initial_guess(x_init, u_init=u_init)

        u_bounds = BoundsOption([u_mi, u_ma], interpolation=InterpolationType.CONSTANT)
        ocp.update_bounds(u_bounds=u_bounds)

        if use_ACADOS:
            sol = ocp.solve(
                solver=Solver.ACADOS,
                show_online_optim=False,
                solver_options={
                    "nlp_solver_max_iter": 100,
                    "nlp_solver_tol_comp": 1e-4,
                    "nlp_solver_tol_eq": 1e-4,
                    "nlp_solver_tol_stat": 1e-4,
                    "integrator_type": "IRK",
                    "nlp_solver_type": "SQP",
                    "sim_method_num_steps": 1,
                })
            # states, controls = Data.get_data(ocp, sol)
            # e = controls['muscles']
            # t = np.linspace(0, T, Ns + 1)
            # for j in range(biorbd_model.nbMuscles()):
            #     plt.subplot(4, 5, j + 1)
            #     plt.step(t, e[j, :])
            # plt.show()
            states, controls = Data.get_data(ocp, sol)
            e_exc = controls['muscles']
            # if i != 0:
                # Take excitations to track
                # update u_init and u_bounds
                # u_init = InitialGuessOption(excitations_init[0], interpolation=InterpolationType.CONSTANT)
                # x_init = InitialGuessOption(np.tile(np.concatenate(
                #     (x0, [0.1] * biorbd_model.nbMuscles())), (ocp.nlp[0].ns + 1, 1)).T,
                #                             interpolation=InterpolationType.EACH_FRAME)
                # ocp.update_initial_guess(x_init, u_init=u_init)
                # u_bounds = BoundsOption([excitations_min[0], excitations_max[0]], interpolation=InterpolationType.CONSTANT)
                # ocp.update_bounds(u_bounds=u_bounds)
                #
                # # Update Objectives
                # objective_functions = ObjectiveList()
                # objective_functions.add(
                #     Objective.Lagrange.MINIMIZE_STATE, weight=1, states_idx=np.array(range(biorbd_model.nbQ()))
                # )
                # objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=10,
                #                         states_idx=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)))
                # objective_functions.add(
                #     Objective.Lagrange.MINIMIZE_STATE,
                #     weight=1,
                #     states_idx=np.array(
                #         range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles()))
                # )
                # objective_functions.add(
                #     Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL,
                #     weight=100,
                #     muscles_idx=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16])
                # )
                # objective_functions.add(
                #     Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL,
                #     weight=100000,
                #     target=e_exc,
                #     muscles_idx=np.array([9, 10, 17, 18]),
                # )
                # objective_functions.add(
                #     Objective.Mayer.TRACK_STATE,
                #     weight=100000,
                #     target=np.tile(xT[:biorbd_model.nbQ()], (e_exc.shape[1] + 1, 1)).T,
                #     states_idx=np.array(range(biorbd_model.nbQ()))
                # )
                # ocp.update_objectives(objective_functions)
                # sol = ocp.solve(
                #     solver=Solver.ACADOS,
                #     show_online_optim=False,
                #     solver_options={
                #         "nlp_solver_max_iter": 30,
                #         "nlp_solver_tol_comp": 1e-4,
                #         "nlp_solver_tol_eq": 1e-4,
                #         "nlp_solver_tol_stat": 1e-4,
                #         "integrator_type": "IRK",
                #         "nlp_solver_type": "SQP",
                #         "sim_method_num_steps": 1,
                #     })
                # states, controls = Data.get_data(ocp, sol)
                # u_co = controls['muscles']
                # t = np.linspace(0, T, Ns + 1)
                # plt.figure("Muscles controls")
                # for i in range(biorbd_model.nbMuscles()):
                #     plt.subplot(4, 5, i + 1)
                #     plt.step(t, u_co[i, :])
                #     plt.step(t, e_exc[i, :], c='red')
                # plt.show()

            if save_data:
                if i == 0:
                    ocp.save_get_data(
                        sol, f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{i}.bob"
                    )
                ocp.save_get_data(
                    sol, f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{i}_tmp.bob"
                )

        # if os.path.isfile(f"solutions/sim_ac_{int(T*1000)}ms_{Ns}sn_{motion}_co_level_{i}.bob"):
            #     ocp.save_get_data(
            #         sol, f"solutions/sim_ac_{int(T*1000)}ms_{Ns}sn_{motion}_co_level_{i}_1.bob"
            #     )
            # else:
            #     ocp.save_get_data(
            #         sol, f"solutions/sim_ac_{int(T*1000)}ms_{Ns}sn_{motion}_co_level_{i}.bob"
            #     )

        if use_IPOPT:
            sol = ocp.solve(
                solver=Solver.IPOPT,
                show_online_optim=False,
                solver_options={
                    "tol": 1e-4,
                    "dual_inf_tol": 1e-4,
                    "constr_viol_tol": 1e-4,
                    "compl_inf_tol": 1e-4,
                    "linear_solver": "ma57",
                    "max_iter": 500,
                    "hessian_approximation": "limited-memory",
                },
            )
            if i != 0:
                # Take excitations to track
                states, controls = Data.get_data(ocp, sol)
                e_exc = controls['muscles']

                # Update Objectives
                objective_functions = ObjectiveList()
                objective_functions.add(
                    Objective.Lagrange.MINIMIZE_STATE, weight=1, states_idx=np.array(range(biorbd_model.nbQ()))
                )
                objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=10,
                                        states_idx=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)))
                objective_functions.add(
                    Objective.Lagrange.MINIMIZE_STATE,
                    weight=1,
                    states_idx=np.array(
                        range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles()))
                )
                objective_functions.add(
                    Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL,
                    weight=10,
                    muscles_idx=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16])
                )
                objective_functions.add(
                    Objective.Lagrange.TRACK_MUSCLES_CONTROL,
                    weight=100,
                    target=e_exc,
                    muscles_idx=np.array([9, 10, 17, 18]),
                )
                objective_functions.add(
                    Objective.Mayer.TRACK_STATE,
                    weight=100000,
                    target=np.tile(xT[:biorbd_model.nbQ()], (e_exc.shape[1] + 1, 1)).T,
                    states_idx=np.array(range(biorbd_model.nbQ()))
                )
                ocp.update_objectives(objective_functions)
                sol = ocp.solve(
                    solver=Solver.IPOPT,
                    show_online_optim=False,
                    solver_options={
                        "tol": 1e-4,
                        "dual_inf_tol": 1e-4,
                        "constr_viol_tol": 1e-4,
                        "compl_inf_tol": 1e-4,
                        "linear_solver": "ma57",
                        "max_iter": 500,
                        "hessian_approximation": "limited-memory",
                    },
                )
            ocp.save_get_data(
                sol, f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{i}.bob"
            )
            if os.path.isfile(f"solutions/sim_ip_{int(T*1000)}ms_{Ns}sn_{motion}_co_level_{i}.bob"):
                ocp.save_get_data(
                    sol, f"solutions/sim_ip_{int(T*1000)}ms_{Ns}sn_{motion}_co_level_{i}_1.bob"
                )
            else:
                ocp.save_get_data(
                    sol, f"solutions/sim_ip_{int(T*1000)}ms_{Ns}sn_{motion}_co_level_{i}.bob"
                )

        if use_BO:
            with open(
                    f"solutions/sim_ac_{int(T*1000)}ms_{Ns}sn_{motion}_co_level_{i}.bob", 'rb'
            ) as file:
                data = pickle.load(file)
            states = data['data'][0]
            controls = data['data'][1]
            q = states['q']
            qdot = states['q_dot']
            a = states['muscles']
            u = controls['muscles']
            # tau = controls['tau']
            t = np.linspace(0, T, Ns + 1)
            q_name = [biorbd_model.nameDof()[i].to_string() for i in range(biorbd_model.nbQ())]
            plt.figure("Q")
            for i in range(q.shape[0]):
                plt.subplot(2, 3, i + 1)
                plt.plot(t, q[i, :], c='purple')
                plt.title(q_name[i])

            plt.figure("Q_dot")
            for i in range(q.shape[0]):
                plt.subplot(2, 3, i + 1)
                plt.plot(t, qdot[i, :], c='purple')
                plt.title(q_name[i])

            # plt.figure("Tau")
            # for i in range(q.shape[0]):
            #     plt.subplot(2, 3, i + 1)
            #     plt.plot(t, tau[i, :], c='orange')
            #     plt.title(biorbd_model.muscleNames()[i].to_string())

            plt.figure("Muscles controls")
            for i in range(u.shape[0]):
                plt.subplot(4, 5, i + 1)
                plt.step(t, u[i, :], c='orange')
                plt.plot(t, a[i, :], c='purple')
                plt.title(biorbd_model.muscleNames()[i].to_string())
            plt.legend(labels=['excitations', "activations"], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.show()

            b = bioviz.Viz(model_path="arm_wt_rot_scap.bioMod")
            b.load_movement(q)
            b.exec()

        # --- Show results --- #
        result = ShowResult(ocp, sol)
        result.graphs()
        # result.animate()

