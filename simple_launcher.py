import numpy as np
from datetime import date
from os import system
from module.save_eco import BackUp


def main():

    system("python3 setup.py build_ext --inplace")
    from module.eco import Launcher

    parameters = {
        "x0": 500,  # np.random.randint(500),
        "x1": 500,  # np.random.randint(500),
        "x2": 500,  # np.random.randint(500),
        "t_max": 1000,  # np.random.randint(10, 50),
        "alpha": 0.5,  # np.random.random(),
        "tau": 0.11,  # np.random.random(),
        "gamma": 0.8,  # np.random.random(),
        "q": 1,  # np.random.random(),
        "eco_idx": np.random.randint(500),  # These last two are for the back up
        "date":  date.today()
    }

    results = Launcher.launch(parameters, single=True)
    BackUp.save_data(results=results, parameters=parameters, backup_file="backup_file")

if __name__ == "__main__":

    main()
