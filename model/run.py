import numpy as np
import tqdm

from . import model
from . import data_structure

############################################################
#            RUN SIMULATION                                #
############################################################


class SimulationRunner:

    def __init__(self, x0, x1, x2, alpha, tau, gamma, t_max, q):

        # Time the simulation should last
        self.t_max = t_max

        self.t = -1

        # Create the economy to simulate
        self.eco = model.Economy(x0=x0, x1=x1, x2=x2, alpha=alpha, tau=tau, gamma=gamma, t_max=t_max, q=q)

        # For pre-analysis
        self.indirect_exchanges = np.zeros((t_max, 3))
        self.direct_exchanges = np.zeros((t_max, 3))

    def run(self, single=True):

        self.eco.setup()

        if single:

            # Run simulation for as time units as required.
            for _ in tqdm.tqdm(range(self.t_max)):

                self.eco.run()
                self.make_some_stats()
                self.t += 1

        else:

            # Run simulation for as time units as required.
            for t in range(self.t_max):

                self.eco.run()
                self.make_some_stats()
                self.t += 1

        return self.export()

    def make_some_stats(self):

        for i, idx in enumerate([self.eco.idx0, self.eco.idx1, self.eco.idx2]):  # 3 types of agents

            ind1 = self.eco.i_choice[idx][:] == 1
            ind2 = self.eco.i_choice[idx][:] == 2
            self.indirect_exchanges[self.t][i] = np.mean(ind1+ind2)

            direct = self.eco.i_choice[idx][:] == 0
            self.direct_exchanges[self.t][i] = np.mean(direct)

    def export(self):

        return data_structure.Results(
            indirect_exchanges=self.indirect_exchanges,
            direct_exchanges=self.direct_exchanges)


class Launcher(object):

    @classmethod
    def launch(cls, **parameters):

        single = parameters.pop("single")

        ##################################
        #   Beginning of the program     #
        ##################################

        # Create a "simulation runner" that will manage the simulation.
        simulation_runner = SimulationRunner(**parameters)

        # Ask the "simulation runner" to launch the simulation.
        return simulation_runner.run(single)

        ############################
        #   End of the program     #
        ############################
