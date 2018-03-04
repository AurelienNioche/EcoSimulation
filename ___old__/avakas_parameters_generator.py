import numpy as np
import pickle
from datetime import date
from os import path, mkdir
import shutil
import re
from random import shuffle
from ___old__.module import Folders


class ParametersGenerator(object):

    def __init__(self):

        self.t_max = 1000

        self.alpha_list = np.arange(0.1, 1.1, 0.1)

        # Be careful in choosing of 'overflow' in exp computation
        self.tau_list = np.arange(0.01, 0.11, 0.01)
        self.gamma_list = np.arange(0., 1., 0.1)
        self.q_information_list = np.arange(0, 1.1, 0.1)
        self.workforce_step = 25
        self.workforce_mini = 50
        self.workforce_maxi = 100

        self.nb_jobs = 1000

        self.date = str(date.today())

    @staticmethod
    def empty_scripts_folder():

        if path.exists(Folders.data):

            response = input("Do you want to remove data folder?")

            while response not in ['y', 'yes', 'n', 'no', 'Y', 'N']:
                response = input("You can only respond by 'yes' or 'no'.")

            print("Proceeding...")

            if response in ['y', 'yes', 'Y']:

                if path.exists(Folders.data):
                    shutil.rmtree(Folders.data)
                print("Data folder has been erased.")
            else:
                print("Data folder has been conserved.")

        else:
            print("Proceeding...")

        print("Remove old scripts and logs...")

        if path.exists(Folders.scripts):
            shutil.rmtree(Folders.scripts)

        print("Old scripts and logs have been removed.")

        if path.exists(Folders.logs):
            shutil.rmtree(Folders.logs)

    @staticmethod
    def create_folders():

        for directory in Folders.list():

            if not path.exists(directory):

                mkdir(directory)

    def generate_workforce_list(self):

        workforce_list = list()
        workforce = np.zeros(3, dtype=int)

        workforce[:] = self.workforce_mini

        possible_w = np.arange(self.workforce_mini, self.workforce_maxi + 0.1, self.workforce_step)
        # for i in possible_w:
        #     workforce[2] = i
        #     workforce_list.append(workforce.copy())
        #
        # workforce_step = 50
        # workforce_mini = 100
        # workforce_maxi = 400
        #
        # workforce[:] = workforce_mini
        #
        # possible_w = np.arange(workforce_mini, workforce_maxi + 0.1, workforce_step)
        # for i in possible_w:
        #     workforce[2] = i
        #     workforce_list.append(workforce.copy())

        for i in possible_w:
            for j in possible_w:
                for k in possible_w:
                    if i <= j <= k:
                        workforce[:] = i, j, k
                        workforce_list.append(workforce.copy())

        print("Length of workforce list:", len(workforce_list))
        return workforce_list

    def generate_parameters_list(self, workforce_list):

        idx = 0
        parameters_list = []

        for workforce in workforce_list:
            for alpha in self.alpha_list:
                for tau in self.tau_list:
                    for q_information in self.q_information_list:

                        if q_information == 0:
                            parameters = \
                                {
                                    "x0": workforce[0],
                                    "x1": workforce[1],
                                    "x2": workforce[2],
                                    "t_max": self.t_max,  # Set the number of time units the simulation will run
                                    "alpha": alpha,
                                    "tau": tau,
                                    "gamma": 1,
                                    "q": q_information,
                                    "eco_idx": idx,  # For saving
                                    "date": self.date  # For saving

                                }
                            parameters_list.append(parameters)
                            idx += 1  # increment idx

                        else:

                            for gamma in self.gamma_list:
                                parameters = \
                                    {
                                        "x0": workforce[0],
                                        "x1": workforce[1],
                                        "x2": workforce[2],
                                        "t_max": self.t_max,  # Set the number of time units the simulation will run
                                        "alpha": alpha,
                                        "tau": tau,
                                        "gamma": gamma,
                                        "q": q_information,
                                        "eco_idx": idx,  # For saving
                                        "date": self.date  # For saving
                                    }
                                parameters_list.append(parameters)
                                idx += 1  # increment idx

        return parameters_list

    def generate_input_parameters(self, parameters_list):

        shuffle(parameters_list)  # For trying to equalize computation times between scripts

        len_sub_part = len(parameters_list) / self.nb_jobs
        rounded_len_sub_part = int(len_sub_part)

        # If there is more tasks than jobs...

        if len_sub_part > 1:

            input_parameters_dict = {}  # Keys will be the ID of the script to be executed

            cursor = 0

            for i in range(self.nb_jobs):
                part = parameters_list[cursor:cursor + rounded_len_sub_part]
                input_parameters_dict[i] = part
                cursor += rounded_len_sub_part

            while cursor < len(parameters_list):

                for i in range(self.nb_jobs):

                    if cursor < len(parameters_list):
                        input_parameters_dict[i].append(parameters_list[cursor])
                        cursor += 1

        # If there is an equal number of tasks and jobs, or less...
        else:

            len_sub_part = 1
            self.nb_jobs = len(parameters_list)

            input_parameters_dict = {}
            for i in range(self.nb_jobs):
                # Input parameters for a job is a list containing a unique element
                input_parameters_dict[i] = [parameters_list[i]]

        return input_parameters_dict, len_sub_part

    @staticmethod
    def save_input_parameters(input_parameters):

        print("Save input parameters...")

        for i in range(len(input_parameters)):
            pickle.dump(input_parameters[i],
                        open("{}/slice_{}.p".format(Folders.input_parameters, i), mode="wb"))

        print("Input parameters saved.")

    def create_scripts(self):

        print("Create scripts...")

        root_file = "{}/simulation_template.sh".format(Folders.macro)
        prefix_output_file = "{}/eco-simulation_".format(Folders.scripts)

        for i in range(self.nb_jobs):

            with open(root_file, 'r') as f:
                content = f.read()

            replaced = re.sub('slice_0', 'slice_{}'.format(i), content)
            replaced = re.sub('eco-simulation_0', 'eco-simulation_{}'.format(i), replaced)

            script_name = "{}{}.sh".format(prefix_output_file, i)

            with open(script_name, 'w') as f:
                f.write(replaced)

        print("Scripts created.")

    def run(self):

        workforce_list = self.generate_workforce_list()
        parameters_list = self.generate_parameters_list(workforce_list=workforce_list)
        input_parameters, len_sub_part = self.generate_input_parameters(parameters_list)

        response = input("Number of jobs: {}; number of tasks per job: {}; "
                         "total number of tasks: {}. \n"
                         "Should I proceed?".format(self.nb_jobs, len_sub_part,
                                                    len(parameters_list)))

        while response not in ['y', 'yes', 'n', 'no', 'Y', 'N']:
            response = input("You can only respond by 'yes' or 'no'.")

        if response in ['y', 'yes', 'Y']:

            self.empty_scripts_folder()
            self.create_folders()
            self.save_input_parameters(input_parameters)
            self.create_scripts()

            print("Done!")

        else:

            print("Process aborted by user.")


def main():

    p = ParametersGenerator()
    p.run()


if __name__ == "__main__":
    main()
