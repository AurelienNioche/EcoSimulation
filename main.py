import argparse
import pickle
import sys
from os import system
from multiprocessing import Pool
import json
from module.save_eco import BackUp

system("python3 setup.py build_ext --inplace")
from module.eco import Launcher


class Runner(object):

    def __init__(self):

        self.backup_file = ""

    def start(self):

        parameters_list, backup_file = self.get_parameters()
        print("Backup file:", backup_file)

        self.backup_file = backup_file

        with open("parameters/avakas.json") as avakas_parameters:
            n_core = json.load(avakas_parameters)["n_core"]
        pool = Pool(processes=n_core)
        pool.map(self.run, parameters_list)

    @staticmethod
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

        print(pickle_file)

        return parameters, pickle_file.split("/")[-1].split(".")[0]

    def run(self, parameters):

        results = Launcher.launch(parameters, single=False)
        BackUp.save_data(results=results, parameters=parameters, backup_file=self.backup_file)


def main():

    if sys.version_info[0] < 3:
        raise Exception("Should use Python 3")

    runner = Runner()
    runner.start()


if __name__ == "__main__":

    main()


