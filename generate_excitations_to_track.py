import biorbd
from time import time
import numpy as np
from casadi import MX, Function
import matplotlib.pyplot as plt
import pickle
import sys

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
    BoundsOption
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
    objectives = ObjectiveList()


    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0].concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )
    x_bounds[0].min[:nbQ*2, 0] = x0[:nbQ*2]
    x_bounds[0].max[:nbQ*2, 0] = x0[:nbQ*2]
    x_bounds[0].min[nbQ * 2:nbQ * 2+nbMT, 0] = [0.1] * nbMT
    x_bounds[0].max[nbQ * 2:nbQ * 2+nbMT, 0] = [1] * nbMT
    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [tau_min] * nbGT + [muscle_min] * biorbd_model.nbMuscleTotal(),
            [tau_max] * nbGT + [muscle_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    # Initial guesses
    x_init = InitialGuessOption(np.tile(x0, (number_shooting_points+1, 1)).T, interpolation=InterpolationType.EACH_FRAME)

    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)+0.1
    u_init = InitialGuessOption(np.tile(u0, (number_shooting_points, 1)).T, interpolation=InterpolationType.EACH_FRAME)
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
        objectives,
        use_SX=use_SX,
        nb_threads=nb_threads,
    )

if __name__ == "__main__":
    save_data = True
    T = 8
    Ns = 800
    motion = 'REACH2'

    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
    mat_content = sio.loadmat(f"solutions/state_to_track_{T}phase.mat")
    states = mat_content['state']
    controls = mat_content['controls']
    q_ref = states[:biorbd_model.nbQ(), :]
    dq_ref = states[biorbd_model.nbQ():biorbd_model.nbQ()*2, :]
    nbGT = 0
    nbMT = biorbd_model.nbMuscleTotal()
    nbQ = biorbd_model.nbQ()

    x0 = np.hstack([q_ref[:, 0], dq_ref[:, 0], [0.3] * biorbd_model.nbMuscles()])

    tau_init = 0
    muscle_init = 0.5
    use_CO = True

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

    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T, x0=x0, nbGT=nbGT, number_shooting_points=Ns, use_SX=True)

    for co in range(len(excitations_init)):
    # for co in range(2, 4):
        u_i = excitations_init[co]
        u_mi = excitations_min[co]
        u_ma = excitations_max

        u_init = InitialGuessOption(np.tile(u_i, (ocp.nlp[0].ns, 1)).T, interpolation=InterpolationType.EACH_FRAME)
        x_init = InitialGuessOption(np.tile(x0, (ocp.nlp[0].ns+1, 1)).T, interpolation=InterpolationType.EACH_FRAME)
        ocp.update_initial_guess(x_init, u_init=u_init)

        u_bounds = BoundsOption([u_mi, u_ma], interpolation=InterpolationType.CONSTANT)
        ocp.update_bounds(u_bounds=u_bounds)

        objectives = ObjectiveList()
        objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=1000)

        objectives.add(
            Objective.Lagrange.TRACK_STATE,
            weight=100000,
            target=states[:biorbd_model.nbQ(), :],
            states_idx=np.array(range(nbQ))
        )

        objectives.add(
            Objective.Lagrange.MINIMIZE_STATE, weight=100, states_idx=np.array(range(nbQ * 2, nbQ * 2 + nbMT))
        )

        objectives.add(
            Objective.Lagrange.MINIMIZE_STATE, weight=1000, states_idx=np.array(range(nbQ, nbQ * 2))
        )
        ocp.update_objectives(objectives)

        tic = time()
        sol = ocp.solve(solver=Solver.ACADOS,
                        show_online_optim=False,
                        solver_options={
                            "nlp_solver_max_iter": 50,
                            "nlp_solver_tol_comp": 1e-6,
                            "nlp_solver_tol_eq": 1e-6,
                            "nlp_solver_tol_stat": 1e-6,
                            "integrator_type": "IRK",
                            "nlp_solver_type": "SQP",
                            "sim_method_num_steps": 1,
                        })
        print(f"Time to solve with ACADOS : {time()-tic} s")
        toc = time() - tic
        print(f"Total time to solve with ACADOS : {toc} s")
        data_est = Data.get_data(ocp, sol)
        X_est = np.vstack([data_est[0]['q'], data_est[0]['q_dot'], data_est[0]['muscles']])
        U_est = data_est[1]['muscles']

        plt.subplot(211)
        plt.plot(X_est[:biorbd_model.nbQ(), :].T, 'x')
        plt.gca().set_prop_cycle(None)
        plt.plot(q_ref.T)
        plt.legend(labels=['Q estimate', 'Q truth'], bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
        plt.subplot(212)
        plt.plot(X_est[biorbd_model.nbQ():biorbd_model.nbQ()*2, :].T, 'x')
        plt.gca().set_prop_cycle(None)
        plt.plot(dq_ref.T)
        plt.legend(labels=['Qdot estimate', 'Qdot truth'], bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
        # plt.tight_layout()
        plt.figure('Muscles excitations')
        for i in range(biorbd_model.nbMuscles()):
            plt.subplot(4, 5, i + 1)
            plt.plot(controls[i, :].T)
            plt.plot(U_est[i, :].T, 'x')
            plt.plot(X_est[nbQ*2 + i, :].T, '--')
            plt.title(biorbd_model.muscleNames()[i].to_string())
        plt.legend(
            labels=['u_ref', 'u_est'], bbox_to_anchor=(1.05, 1), loc='upper left',
            borderaxespad=0.
        )
        plt.show()
        if save_data:
            if co == 0:
                ocp.save_get_data(
                    sol, f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{co}.bob"
                )
            ocp.save_get_data(
                sol, f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{co}_tmp.bob"
            )
        # --- Show results --- #
        # result = ShowResult(ocp, sol)
        # result.graphs()
        # result.animate()
