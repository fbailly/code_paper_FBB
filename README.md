# code_paper_FBB

Code repo for Bailly et al., *Real-Time and Dynamically Consistent Estimation of Muscle Forces Using a Moving Horizon EMG-Marker Tracking Algorithm*, 2021.
This piece of code depends on the [*bioptim*](https://github.com/pyomeca/bioptim) Python librairy. It is compatible with its first release [*v1.0.0*](https://github.com/pyomeca/bioptim/tree/TheRealDebut).

# How to use ?

We recommend you install *bioptim* and *acados* following these [instructions](https://github.com/pyomeca/bioptim#how-to-install).
Then you can run *main_script.py* by adjusting the flags at the beggining of the code.
For a first use, you will need to generate the dataset used in the paper.
To do so :
```
"load_data": False,
```
If you don't want to generate the full dataset (it would take several hours), you can adjust the following values of the ```var``` dictionnary :
```
    "nb_try": 30, # number of trials per condition
    "marker_noise_lvl": [0, 0.002, 0.005, 0.01],
    "EMG_noise_lvl": [0, 1, 1.5, 2],
    "nb_co_lvl": 4,
```
