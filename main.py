import argparse
import pickle
import sys
from multiprocessing import Pool
from module.eco import Launcher
from module.save_eco import BackUp


def get_parameters():

    parser = argparse.ArgumentParser()

    parser.add_argument('parameters', type=str,
                        help='A name of pickle file for parameters is required!')

    args = parser.parse_args()

    pickle_file = args.parameters
    try:
        parameters = pickle.load(open(pickle_file, mode='rb'))
    except Exception as e:
        print("Problems in loading", pickle_file)
        raise e

    return parameters


def run(parameters):

    results = Launcher.launch(parameters, single=False)
    BackUp.save_data(results=results, parameters=parameters)


def main():
    if sys.version_info[0] < 3:
        raise Exception("Should use Python 3")

    parameters_list = get_parameters()

    p = Pool()
    p.map(run, parameters_list)


if __name__ == "__main__":

    main()


