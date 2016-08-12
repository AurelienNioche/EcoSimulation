from pylab import np, plt
import matplotlib.gridspec as gridspec
import pickle


class Analyst(object):

    def __init__(self):

        self.last = open("results/last.txt").read()
        self.parameters = pickle.load(open("results/parameters_{}.p".format(self.last), mode='rb'))
        self.main_results = pickle.load(open("results/main_results_{}.p".format(self.last), mode='rb'))
        self.i_choice = np.load("results/i_choice_{}.npy".format(self.last))
        self.received_information = np.load("results/received_information_{}.npy".format(self.last))
        self.finding_a_partner = np.load("results/finding_a_partner_{}.npy".format(self.last))

        self.workforce = np.array([self.parameters["x0"], self.parameters["x1"], self.parameters["x2"]])
        self.n = np.sum(self.workforce)  # Total number of agents

        self.type = np.zeros(self.n, dtype=int)

        self.type[:] = np.concatenate(([0, ] * self.workforce[0],
                                       [1, ] * self.workforce[1],
                                       [2, ] * self.workforce[2]))

        self.idx = []
        for i in range(3):

            self.idx.append(np.where(self.type[:] == i)[0])

        self.t_max = self.parameters["t_max"]

        # ---- #

        # To convert relative choices into absolute choices
        # 0 : 0 -> 1
        # 1 : 0 -> 2
        # 2 : 1 -> 0
        # 3 : 1 -> 2
        # 4 : 2 -> 0
        # 5 : 2 -> 1

        # i: type; j: i_choice
        self.absolute_choice_matrix = np.array([
            [0, 1, 5, 4],
            [3, 2, 1, 0],
            [4, 5, 2, 3]], dtype=int)

        # ---- #

        self.success_by_absolute_choice = np.zeros((self.t_max, 6))  # 6 is number of absolute choices : 0->1, etc.

        self.choice_frequency = np.zeros((self.t_max, 4),
                                         dtype=[("type0", float, 1),
                                                ("type1", float, 1),
                                                ("type2", float, 1)])

        self.mean_received_information = np.zeros((self.t_max, 4),
                                                  dtype=[("type0", float, 1),
                                                         ("type1", float, 1),
                                                         ("type2", float, 1)])

    def print_information(self):

        print("Session:", self.last)
        print("Main results:", self.main_results)
        # np.savetxt("output.txt", self.i_choice)
        # print(len(self.i_choice[0][self.idx[0]]))

    def compute_reward_frequencies(self):

        # Give per type the actual number of success

        for t in range(self.t_max):

            absolute_choice = np.zeros(self.n)
            absolute_choice[:] = self.compute_absolute_choices(self.type, self.i_choice[t])

            for exchange_type in range(6):  # 6 is number of absolute choices : 0->1, etc.

                bool_absolute_choice = absolute_choice[:] == exchange_type
                if np.sum(bool_absolute_choice) > 0:
                    self.success_by_absolute_choice[t, exchange_type] = \
                        np.mean(self.finding_a_partner[t][bool_absolute_choice])
                else:
                    self.success_by_absolute_choice[t, exchange_type] = -1

    def compute_absolute_choices(self, agent_type, agent_choice):

        return self.absolute_choice_matrix[agent_type, agent_choice]

    def compute_information(self):

        for t in range(self.t_max):

            for exchange_type in range(4):

                for agent_type in range(3):
                # agent_type = 1
                #print("t", self.i_choice[t][self.idx[agent_type]])

                    info = self.received_information[t][self.idx[agent_type], exchange_type]
                    info = info[info != -1]
                    if np.sum(info != -1) > 0:
                        mean_info = np.mean(info)
                    else:
                        mean_info = -1

                    self.mean_received_information[t, exchange_type]["type{}".format(agent_type)] = \
                        mean_info
        # print(self.mean_received_information)



    # cdef
    # compute_noise(self, cnp.ndarray
    # array):
    # #
    # #     cdef:
    # #         float mean, noise
    # #
    # #     if array.size:
    # #         mean = np.mean(array)
    # #         if mean != 0 and mean != 1:
    # #             noise = -np.log2(1 - mean) + mean * np.log2((1 - mean) / mean)
    # #         else:
    # #             noise = 0
    # #     else:
    # #         noise = -1.
    # #
    # #     return noise

    def compute_choice_frequencies(self):

        for t in range(self.t_max):

            for exchange_type in range(4):

                for agent_type in range(3):
                # agent_type = 1
                #print("t", self.i_choice[t][self.idx[agent_type]])

                    self.choice_frequency[t, exchange_type]["type{}".format(agent_type)] = \
                        np.mean(self.i_choice[t][self.idx[agent_type]] == exchange_type)

    def plot_choice_frequencies(self):

        plt.figure("Choice frequencies", figsize=(20, 13))

        x = np.arange(self.t_max)

        sub = [411, 412, 413, 414]
        colors = ["red", "blue", "green"]

        linewidth = 2

        for exchange_type in range(4):

            plt.subplot(sub[exchange_type])

            for agent_type in range(3):

                plt.plot(x, self.choice_frequency[:, exchange_type]["type{}".format(agent_type)],
                         c=colors[agent_type], linewidth=linewidth)

        plt.show()

    def plot_for_pigeon(self):

        plt.figure("Pigeon plot", figsize=(23, 14))

        G = gridspec.GridSpec(4, 3)

        # t_max = self.t_max
        t_max = self.t_max
        x = np.arange(t_max)
        linewidth = 2

        agent_types = np.arange(3)
        for agent_type in agent_types:
            for exchange_type in range(4):

                index_for_subplot = 4*agent_type+exchange_type
                print(index_for_subplot)

                plt.subplot(G[exchange_type, agent_type])

                plt.plot(x, self.choice_frequency[np.arange(t_max),
                                                  exchange_type]["type{}".format(agent_type)],
                         c="red", linewidth=linewidth)

                plt.plot(x, self.success_by_absolute_choice[np.arange(t_max),
                                                            self.absolute_choice_matrix[agent_type, exchange_type]],
                         c="green", linewidth=linewidth, linestyle="-")

                plt.plot(x, self.mean_received_information[np.arange(t_max),
                                                           exchange_type]["type{}".format(agent_type)],
                         c="blue", linewidth=linewidth, linestyle="--")

        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.show()




def main():

    a = Analyst()
    a.print_information()
    a.compute_information()
    a.compute_reward_frequencies()
    a.compute_choice_frequencies()
    a.plot_for_pigeon()
    # a.plot_choice_frequencies()


if __name__ == "__main__":

    main()