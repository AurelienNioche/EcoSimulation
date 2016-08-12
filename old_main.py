import numpy as np
from multiprocessing import cpu_count
from module.simulation_manager import SimulationManager
from module.eco import SimulationRunner


class Launcher(object):

    @classmethod
    def launch(cls, parameters):

        ##################################
        #   Beginning of the program     #
        ##################################

        # Create a "simulation runner" that will manage the simulation.
        simulation_runner = SimulationRunner(parameters)

        # Ask the "simulation runner" to launch the simulation.
        return simulation_runner.run()

        ############################
        #   End of the program     #
        ############################

    @classmethod
    def simple_run(cls):

        param = \
            {
                "workforce": np.array([10, 10, 10], dtype=int),
                "alpha": 0.4,  # Set the coefficient learning.
                "tau": 0.03,  # Set the softmax parameter.
                "time_limit": 300,  # Set the number of time units the simulation will run.
                "welfare_delta": 100,  # Set the number of time units you want for analysing welfare.
                "money_threshold": .90,  # Set the 'money threshold'
                "money_delta": 100,  # Set the number of time units to use for check
                                     # if the economy is in a monetary state.
                "q_information": 1,  # Set the quantity of information learned
                                     # by each agent at each round.
                "epsilon": 0.2,  # Set the importance of other agents'results in
                                 # front of an individual result for a given agent.
            }

        # np.save('param.npy', param)
        result = cls.launch(param)
        print(result)

    @classmethod
    def multiple_run(cls):

        arg_list = []

        for alpha in np.arange(0.1, 0.2, 0.2):
            for tau in np.arange(0.01, 0.04, 0.02):
                for q_info in np.arange(0, 1.1, 0.2):
                    for epsilon in np.arange(0, 1.1, 0.25):

                        for i in np.arange(100, 301, 100):
                            for j in np.arange(100, 301, 100):
                                for k in np.arange(100, 301, 100):
                                    if i <= j <= k:
                                        param = \
                                            {
                                                "workforce": np.array([i, j, k], dtype=int),
                                                "alpha": alpha,  # Set the coefficient learning.
                                                "tau": tau,  # Set the softmax parameter.
                                                "time_limit": 1000,  # Set the number of time units the
                                                # simulation will run.
                                                "welfare_delta": 100,  # Set the number of time units you want for
                                                # analysing welfare.
                                                "money_threshold": .90,  # Set the 'money threshold'
                                                "money_delta": 100,  # Set the number of time units to use for check
                                                # if the economy is in a monetary state.
                                                "q_information": q_info,  # Set the quantity of information learned
                                                # by each agent at each round.
                                                "epsilon": epsilon,  # Set the importance of other agents'results in
                                                # front of an individual result for a given agent.
                                            }

                                        arg_list.append(param)

        s = SimulationManager(n_workers=cpu_count()-1, back_up_fq=10)
        s.launch_test(function=cls.launch, arg_list=arg_list)


if __name__ == "__main__":

    Launcher.simple_run()



