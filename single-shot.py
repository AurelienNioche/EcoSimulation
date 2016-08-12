import numpy as np
from os.path import exists
from os import mkdir
from datetime import datetime
from module.eco import Launcher
from module.save_eco import BackUp


def date():

    return str(datetime.now())[:-10].replace(" ", "_").replace(":", "-")


def simple_run(logs=True):

    t_max = 1000
    workforce = np.array([200, 200, 200], dtype=int)
    tau = 0.01
    alpha = 0.3
    epsilon = 0.3
    q_information = 0.8

    root_folder = "../single_shot_data"

    if not exists(root_folder):

        mkdir(root_folder)

    if logs:

        logs_folder = "../single_shot_logs"
        if not exists(logs_folder):
            mkdir(logs_folder)

    param = \
        {
            "workforce": workforce,
            "t_max": t_max,  # Set the number of time units the simulation will run.
            "alpha": alpha,
            "tau": tau,
            "epsilon": epsilon,
            "q_information": q_information,
            "date": date(),
            "idx": np.random.randint(99999)
        }

    results = Launcher.launch(param, single=True)
    BackUp.save_data(results=results, parameters=param, root_folder=root_folder)


if __name__ == "__main__":

    simple_run(logs=False)
