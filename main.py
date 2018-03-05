import os
import json
import pickle
import argparse
import multiprocessing as mlt
import tqdm
from pylab import np, plt

import model.run
import model.data_structure
import analysis.pool.run


def produce_pool_data():

    with open("parameters/pool.json", "r") as f:
        pool_param = json.load(f)

    np.random.seed(pool_param["seed"])

    parameters = []
    for _ in range(pool_param["n"]):

        x = np.random.choice(pool_param["x"], size=3)
        x = np.sort(x)

        parameters.append({
          "x0": x[0],
          "x1": x[1],
          "x2": x[2],
          "gamma": np.random.uniform(pool_param["gamma_min"], pool_param["gamma_max"]),
          "q": np.random.uniform(pool_param["q_min"], pool_param["q_max"]),
          "alpha": np.random.uniform(pool_param["alpha_min"], pool_param["alpha_max"]),
          "tau": np.random.uniform(pool_param["tau_min"], pool_param["tau_max"]),
          "t_max": pool_param["t_max"],
          "seed": np.random.randint(0, 2**32-1),
          "single": False
        })

    pool = mlt.Pool()

    backups = []

    for bkp in tqdm.tqdm(
            pool.imap_unordered(model.run.run, parameters),
            total=len(parameters)):
        backups.append(bkp)

    pool_r = model.data_structure.PoolResults(
        parameters=pool_param,
        results=backups
    )

    os.makedirs("data", exist_ok=True)
    with open("data/pool.p", "wb") as f:
        pickle.dump(pool_r, f)

    return pool_r


def main():

    if not os.path.exists("data/pool.p"):
        data = produce_pool_data()

    else:
        with open("data/pool.p", "rb") as f:
            data = pickle.load(f)

    analysis.pool.run.run(data)


def main_single():

    with open("parameters/single.json", "r") as f:
        parameters = json.load(f)

    r = model.run.run(parameters)

    plt.plot(r.indirect_exchanges)
    plt.show()


if __name__ == "__main__":

    # Parse the arguments given in command line
    parser = argparse.ArgumentParser(description='Produce figures.')
    parser.add_argument('-s', '--single', action="store_true", default=False,
                        help="Run a single simulation")
    parsed_args = parser.parse_args()

    if parsed_args.single:
        main_single()

    else:
        main()
