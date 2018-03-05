import numpy as np
import tqdm

from . import model
from . import data_structure

############################################################
#            RUN SIMULATION                                #
############################################################


class SimulationRunner:

    def __init__(self, x0, x1, x2, alpha, tau, gamma, t_max, q, seed, single):

        # Pool simulation or single one
        self.single = single

        # Time the simulation should last and current time
        self.t_max = t_max
        self.t = 0

        # Create the economy to simulate
        self.eco = model.Economy(x0=x0, x1=x1, x2=x2, alpha=alpha, tau=tau, gamma=gamma, t_max=t_max, q=q, seed=seed)

        # For pre-analysis
        self.indirect_exchanges = np.zeros((t_max, 3))
        self.direct_exchanges = np.zeros((t_max, 3))

    def run(self):

        self.eco.setup()

        iterable = tqdm.tqdm(range(self.t_max)) if self.single else range(self.t_max)

        # Run simulation for as time units as required.
        for _ in iterable:

            self.eco.run()
            self.make_some_stats()
            self.t += 1

        return self.direct_exchanges, self.indirect_exchanges

    def make_some_stats(self):

        for i, idx in enumerate([self.eco.idx0, self.eco.idx1, self.eco.idx2]):  # 3 types of agents

            ind1 = self.eco.i_choice[idx][:] == 1
            ind2 = self.eco.i_choice[idx][:] == 2
            self.indirect_exchanges[self.t][i] = np.mean(ind1+ind2)

            direct = self.eco.i_choice[idx][:] == 0
            self.direct_exchanges[self.t][i] = np.mean(direct)


def run(parameters):

    ##################################
    #   Beginning of the program     #
    ##################################

    # Create a "simulation runner" that will manage the simulation.
    simulation_runner = SimulationRunner(**parameters)

    # Ask the "simulation runner" to launch the simulation.
    direct_exchanges, indirect_exchanges = simulation_runner.run()

    # Format data
    return data_structure.Results(
        parameters=parameters,
        indirect_exchanges=indirect_exchanges,
        direct_exchanges=direct_exchanges
    )

    ############################
    #   End of the program     #
    ############################
