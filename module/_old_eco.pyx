import numpy as np
# from time import time
cimport numpy as cnp

############################################
#           NOTATION                       #
############################################

# For the needs of coding, we don't use systematically here the same notation as in the article.
# Here are the matches:

# For an object:
# 'i' means a production good;
# 'j' means a consumption good;
# 'k' means the third good.

# For agent type:
# * '0' means a type-12 agent;
# * '1' means a type-22 agent;
# * '2' means a type-31 agent.

# For a decision:
# * '0' means 'type-i decision';
# * '1' means 'type-k decision'.

# For a choice:
# * '0' means 'ij' if the agent faces a type-i decision and 'kj' if the agent faces a type-k decision;
# * '1' means 'ik'  if the agent faces a type-i decision and 'ki'  if the agent faces type-k decision.

# For markets,
# * '0' means the part of the market '12' where are the agents willing
#       to exchange type-1 good against type-2 good;
# * '1' means the part of the market '12' where are the agents willing
#       to exchange type-2 good against type-1 good;
# * '2' means the part of the market '23' where are the agents willing
#       to exchange type-2 good against type-3 good;
# * '3' means the part of the market '23' where are the agents willing
#       to exchange type-3 good against type-2 good;
# * '4' means the part of the market '31' where are the agents willing
#       to exchange type-3 good against type-1 good;
# * '5' means the part of the market '31' where are the agents willing
#       to exchange type-1 good against type-3 good.


cdef class Economy(object):

    cdef:
        int money_delta, welfare_delta, time_limit
        double money_threshold
        cnp.ndarray type, value_option0, value_option1, probability_of_choosing_option0, random_number, choice, \
            place, placement, decision_array, finding_a_partner, decision, preferences, \
            value_ij, value_ik, value_kj, value_ki, estimation_ij, estimation_ik, \
            estimation_kj, estimation_ki, i_choice, money_round, money_time, \
            beginning_reward, end_reward, idx0, idx1, idx2, idx3, idx4, idx5, choice_transition
        public int money, t_emergence, t_beginning_reward, t_end_reward, q_information, t, n
        public double alpha, temperature, epsilon
        public cnp.ndarray workforce, beginning_welfare, end_welfare, total_received_information, noise_ij, noise_ik,\
            noise_kj, noise_ki, distortion_ij, distortion_ik, distortion_kj, distortion_ki, noise_ij_error, \
            noise_ik_error, noise_kj_error, noise_ki_error, distortion_ij_error, distortion_ik_error, \
            distortion_kj_error, distortion_ki_error, choice_agent0, choice_agent1, \
            choice_agent2, results, informers_results_mean_for_agent0, informers_results_mean_for_agent1,\
            informers_results_mean_for_agent2
        # public object dico

    def __cinit__(self, object parameters):

        # self.dico = dict()
        # self.dico['initialize_estimation'] = [0, 0]
        # self.dico['update_decision'] = [0, 0]
        # self.dico['update_options_values'] = [0, 0]
        # self.dico['make_a_choice'] = [0, 0]
        # self.dico['who_is_where'] = [0, 0]
        # self.dico['make_the_transactions'] = [0, 0]
        # self.dico['update_estimations'] = [0, 0]
        # self.dico['test_for_money'] = [0, 0]
        # self.dico['determine_money'] = [0, 0]
        # self.dico['compute_welfare'] = [0, 0]
        # self.dico['summarize_welfare'] = [0, 0]

        self.choice_transition = np.array([
            [0, 1, 2, 3],
            [-1, -1, 1, 0],
            [3, 2, -1, -1]])

        self.n = np.sum(parameters["workforce"])  # Total number of agents
        self.workforce = np.zeros(len(parameters["workforce"]), dtype=int)
        self.workforce[:] = parameters["workforce"]  # Number of agents by type
        self.time_limit = parameters["time_limit"]

        self.choice_agent0 = np.zeros([self.time_limit, 4])
        self.choice_agent1 = np.zeros([self.time_limit, 4])
        self.choice_agent2 = np.zeros([self.time_limit, 4])

        self.results = np.zeros([self.time_limit, 6])
        self.informers_results_mean_for_agent0 = np.zeros([self.time_limit, 4])
        self.informers_results_mean_for_agent1 = np.zeros([self.time_limit, 4])
        self.informers_results_mean_for_agent2 = np.zeros([self.time_limit, 4])

        self.alpha = parameters["alpha"]  # Learning coefficient
        self.temperature = parameters["tau"]  # Softmax parameter

        self.money_threshold = parameters["money_threshold"]
        self.money_delta = parameters["money_delta"]
        self.welfare_delta = parameters["welfare_delta"]
        self.q_information = np.round(parameters["q_information"]*self.n)
        if self.q_information == self.n:
            self.q_information -= 1
        self.total_received_information = np.zeros(self.n, dtype=int)
        self.epsilon = parameters["epsilon"]

        self.type = np.zeros(self.n, dtype=int)

        self.type[:] = np.concatenate(([0, ]*self.workforce[0],
                                       [1, ]*self.workforce[1],
                                       [2, ]*self.workforce[2]))

        self.noise_ij = np.zeros([self.time_limit, 3])
        self.noise_ik = np.zeros([self.time_limit, 3])
        self.noise_kj = np.zeros([self.time_limit, 3])
        self.noise_ki = np.zeros([self.time_limit, 3])

        self.distortion_ij = np.zeros([self.time_limit, 3])
        self.distortion_ik = np.zeros([self.time_limit, 3])
        self.distortion_kj = np.zeros([self.time_limit, 3])
        self.distortion_ki = np.zeros([self.time_limit, 3])

        self.noise_ij_error = np.zeros([self.time_limit, 3])
        self.noise_ik_error = np.zeros([self.time_limit, 3])
        self.noise_kj_error = np.zeros([self.time_limit, 3])
        self.noise_ki_error = np.zeros([self.time_limit, 3])

        self.distortion_ij_error = np.zeros([self.time_limit, 3])
        self.distortion_ik_error = np.zeros([self.time_limit, 3])
        self.distortion_kj_error = np.zeros([self.time_limit, 3])
        self.distortion_ki_error = np.zeros([self.time_limit, 3])

        # Each agent possesses an index by which he can be identified.
        #  Here are the the indexes lists corresponding to each type of agent:

        self.idx0 = np.where(self.type == 0)[0]
        self.idx1 = np.where(self.type == 1)[0]
        self.idx2 = np.where(self.type == 2)[0]

        #  The "placement array" is a 3-D matrix (d1: type, d2: decision, d3: choice).
        #  Allow us to retrieve the market where is supposed to go an agent according to:
        #  * his type,
        #  * the decision he faced,
        #  * the choice he made.

        self.placement = np.array(
            [[[0, 5],
              [3, 4]],
             [[2, 1],
              [5, 0]],
             [[4, 3],
              [1, 2]]
            ])

        self.place = np.zeros(self.n, dtype=int)

        # The "decision array" is a 3D-matrix (d1: finding_a_partner, d2: decision, d3: choice).
        # Allow us to retrieve the decision faced by an agent at t according to
        #  * the fact that he succeeded in his exchange at t-1,
        #  * the decision he faced at t-1,
        #  * the choice he made at t-1.
        self.decision_array = np.array(
            [[[0, 0],
              [1, 1]],
             [[0, 1],
              [0, 0]]])

        self.decision = np.zeros(self.n, dtype=int)

        self.choice = np.zeros(self.n, dtype=int)

        self.random_number = np.zeros(self.n, dtype=float)  # Used for taking a decision

        self.probability_of_choosing_option0 = np.zeros(self.n, dtype=float)

        self.finding_a_partner = np.zeros(self.n, dtype=int)

        self.i_choice = np.zeros(self.n, dtype=int)

        # Values for each option of choice.
        # The 'option0' and 'option1' are just the options that are reachable by the agents at time t,
        #  among the four other options.
        self.value_ij = np.zeros(self.n)
        self.value_ik = np.zeros(self.n)
        self.value_kj = np.zeros(self.n)
        self.value_ki = np.zeros(self.n)
        self.value_option0 = np.zeros(self.n)
        self.value_option1 = np.zeros(self.n)

        # Initialize the estimations of easiness of each agents and for each type of exchange.
        self.estimation_ik = np.zeros(self.n)
        self.estimation_ij = np.zeros(self.n)
        self.estimation_kj = np.zeros(self.n)
        self.estimation_ki = np.zeros(self.n)

        # This is the initial guest (same for every agent).
        # '1' means each type of exchange can be expected to be realized in only one unit of time
        # The more the value is close to zero, the more an exchange is expected to be hard.

        self.t = 0

        self.t_emergence = -1

        self.money = -1

        # Number of rounds, a good serve as money
        self.money_round = np.zeros(len(self.workforce))
        self.money_time = np.zeros(len(self.workforce))

        self.t_beginning_reward = 0
        self.t_end_reward = 0
        self.beginning_reward = np.zeros((self.welfare_delta, self.n), dtype=int)
        self.end_reward = np.zeros((self.welfare_delta, self.n), dtype=int)
        self.beginning_welfare = np.zeros(len(self.workforce))
        self.end_welfare = np.zeros(len(self.workforce))

    cpdef setup(self):

        # a = np.random.randint(400)
        # # print("Seed for random estimation :", a)
        # np.random.seed(a)
        self.estimation_ij[:] = self.initialize_estimations()
        self.estimation_ik[:] = self.initialize_estimations()
        self.estimation_kj[:] = self.initialize_estimations()
        self.estimation_ki[:] = self.initialize_estimations()

    cdef initialize_estimations(self):

        cdef:
            int i
            cnp.ndarray estimations

        # a = time()
        estimations = np.random.random(self.n)
        # b = time()
        # self.dico['initialize_estimation'][0] += (b-a)
        # self.dico['initialize_estimation'][1] += 1
        return estimations



    def run(self):

        # Make agents updating the decision they are facing
        self.update_decision()

        # Make agents updating the values they attribute to options
        self.update_options_values()

        # Make agents choosing
        self.make_a_choice()

        # Compute the frequency of choices
        # self.compute_choice_frequency()

        # Compute the frequency of the results for an absolute type of exchange
        # self.compute_results_frequency()

        # Move the agents where they are supposed to go
        self.who_is_where()

        # Realize the transactions in the different markets
        self.make_the_transactions()

        # Make agents learn about the success rates of each type of exchange knowing the results of other agents
        # and themselves.
        self.update_estimations()

        # Test for money
        self.test_for_money()

        self.t += 1

    cdef update_decision(self):

        # a = time()

        # Set the decision each agent faces at time t, according to the fact he succeeded or not in his exchange at t-1,
        #  the decision he previously faced, and the choice he previously made.
        self.decision[:] = self.decision_array[self.finding_a_partner,
                                               self.decision,
                                               self.choice]
        # b = time()
        # self.dico['update_decision'][0] += (b-a)
        # self.dico['update_decision'][1] += 1

    cdef update_options_values(self):

        # a = time()

        # Each agent try to minimize the time to consume
        # That is v(option) = 1/(1/estimation)

        # Set value to each option choice

        self.value_ij[:] = self.estimation_ij
        self.value_kj[:] = self.estimation_kj

        for i in range(self.n):

            if not (self.estimation_ik[i] + self.estimation_kj[i]) == 0:

                self.value_ik[i] = \
                    (self.estimation_ik[i] * self.estimation_kj[i]) / \
                    (self.estimation_ik[i] + self.estimation_kj[i])
            else:  # Avoid division by 0
                self.value_ik[i] = 0

            if not (self.estimation_ki[i] + self.estimation_ij[i]) == 0:
                self.value_ki[i] = \
                    (self.estimation_ki[i] * self.estimation_ij[i]) / \
                    (self.estimation_ki[i] + self.estimation_ij[i])
            else:  # Avoid division by 0
                self.value_ki[i] = 0

        # assert np.max(self.value_ij)<=1
        # assert np.max(self.value_ik)<=1
        # assert np.max(self.value_kj)<=1
        # assert np.max(self.value_ki)<=1
        # assert np.min(self.value_ij)>=0
        # assert np.min(self.value_ik)>=0
        # assert np.min(self.value_kj)>=0
        # assert np.min(self.value_ki)>=0

        # b = time()
        # self.dico['update_options_values'][0] += (b-a)
        # self.dico['update_options_values'][1] += 1

    cdef make_a_choice(self):

        # a = time()

        cdef:
            cnp.ndarray id0, id1

        id0 = np.where(self.decision == 0)[0]
        id1 = np.where(self.decision == 1)[0]

        self.value_option0[id0] = self.value_ij[id0]
        self.value_option1[id0] = self.value_ik[id0]

        self.value_option0[id1] = self.value_kj[id1]
        self.value_option1[id1] = self.value_ki[id1]

        # Set a probability to current option 0 using softmax rule
        # (As there is only 2 options each time, computing probability for a unique option is sufficient)

        self.probability_of_choosing_option0[:] = \
            np.exp(self.value_option0/self.temperature) / \
            (np.exp(self.value_option0/self.temperature) +
             np.exp(self.value_option1/self.temperature))

        self.random_number[:] = np.random.uniform(0., 1., self.n)  # Generate random numbers

        # Make a choice using the probability of choosing option 0 and a random number for each agent
        # Choose option 1 if random number > or = to probability of choosing option 0,
        #  choose option 0 otherwise
        self.choice[:] = self.random_number >= self.probability_of_choosing_option0
        self.i_choice[:] = (self.decision * 2) + self.choice

        # b = time()
        # self.dico['make_a_choice'][0] += (b-a)
        # self.dico['make_a_choice'][1] += 1

    cdef who_is_where(self):

        # a = time()

        # Place the agents according to their type, decision and choice
        self.place[:] = self.placement[self.type, self.decision, self.choice]

        # b= time()
        # self.dico['who_is_where'][0] += (b-a)
        # self.dico['who_is_where'][1] += 1

    cdef make_the_transactions(self):

        # a = time()

        cdef:
            cnp.ndarray ipp0, ipp1, ipp2, ipp3, ipp4, ipp5, ip0, ip1

        # Re-initialize the variable for succeeded exchanges
        self.finding_a_partner[:] = 0

        # Find the attendance of each part of the markets
        ipp0 = np.where(self.place == 0)[0]
        ipp1 = np.where(self.place == 1)[0]
        ipp2 = np.where(self.place == 2)[0]
        ipp3 = np.where(self.place == 3)[0]
        ipp4 = np.where(self.place == 4)[0]
        ipp5 = np.where(self.place == 5)[0]

        # Make as encounters as possible
        for ip0, ip1 in [(ipp0, ipp1), (ipp2, ipp3), (ipp4, ipp5)]:  # Consider the two parts of each market

            # If there is nobody in this particular market, do not do nothing.
            if len(ip0) == 0 or len(ip1) == 0:

                pass

            # If there is less agents in one part of the market than in the other:
            #  * agents in the less attended part get successful (that is they can proceed to an exchange);
            #  * among the agent present in the most attended part, randomly select as agents in that part of the market
            #      that there is on the other market: these selected agents can proceed to their exchange.
            elif len(ip0) < len(ip1):

                self.finding_a_partner[ip0] = 1
                np.random.shuffle(ip1)
                self.finding_a_partner[ip1[:len(ip0)]] = 1

            else:

                self.finding_a_partner[ip1] = 1
                np.random.shuffle(ip0)
                self.finding_a_partner[ip0[:len(ip1)]] = 1

        # b = time()
        # self.dico['make_the_transactions'][0] += (b-a)
        # self.dico['make_the_transactions'][1] += 1

    # cdef compute_choice_frequency(self):
    #
    #     cdef:
    #         int i
    #
    #     # cdef:
    #     #     list idx_tot, choice, choice_frequency
    #     #     cnp.ndarray idx_choice
    #
    #     # idx_tot = [self.idx0, self.idx1, self.idx2]
    #     # choice = []
    #     # choice_frequency = [[0, 0, 0, 0],
    #     #                     [0, 0, 0, 0],
    #     #                     [0, 0, 0, 0]]
    #     #
    #     # for i in range(3):
    #     #
    #     #     choice += [np.zeros(len(idx_tot[i]))]
    #     #     choice[i][:] = self.i_choice[idx_tot[i]]
    #     #
    #     #     for j in range(4):
    #     #
    #     #         idx_choice = np.where(choice[i] == j)[0]
    #     #         if idx_choice.size:
    #     #             choice_frequency[i][j] = float(len(idx_choice))/float(len(choice[i]))
    #     #         else:
    #     #             choice_frequency[i][j] = 0
    #
    #     for i in range(4):  # There are four possible choices
    #
    #         self.choice_agent0[self.t, 0] = np.mean(self.i_choice[self.idx0][:] == 0)
    #         self.choice_agent0[self.t, 1] = np.mean(self.i_choice[self.idx0][:] == 1)
    #         self.choice_agent0[self.t, 2] = np.mean(self.i_choice[self.idx0][:] == 2)
    #         self.choice_agent0[self.t, 3] = np.mean(self.i_choice[self.idx0][:] == 3)
    #
    #         self.choice_agent1[self.t, 0] = np.mean(self.i_choice[self.idx1][:] == 0)
    #         self.choice_agent1[self.t, 1] = np.mean(self.i_choice[self.idx1][:] == 1)
    #         self.choice_agent1[self.t, 2] = np.mean(self.i_choice[self.idx1][:] == 2)
    #         self.choice_agent1[self.t, 3] = np.mean(self.i_choice[self.idx1][:] == 3)
    #
    #         self.choice_agent2[self.t, 0] = np.mean(self.i_choice[self.idx2][:] == 0)
    #         self.choice_agent2[self.t, 1] = np.mean(self.i_choice[self.idx2][:] == 1)
    #         self.choice_agent2[self.t, 2] = np.mean(self.i_choice[self.idx2][:] == 2)
    #         self.choice_agent2[self.t, 3] = np.mean(self.i_choice[self.idx2][:] == 3)
    #
    # cdef compute_results_frequency(self):
    #
    #     cdef:
    #         list idx0, idx1, idx2, idx3, idx4, idx5, idx_type_of_good
    #         cnp.ndarray choices0, choices1, choices2
    #         int i
    #
    #     choices0 = np.zeros(self.n)-1.
    #     choices1 = np.zeros(self.n)-1.
    #     choices2 = np.zeros(self.n)-1.
    #     choices0[self.idx0] = self.i_choice[self.idx0]
    #     choices1[self.idx1] = self.i_choice[self.idx1]
    #     choices2[self.idx2] = self.i_choice[self.idx2]
    #
    #     idx0 = list(np.where(choices0 == 0)[0]) + list(np.where(choices1 == 3)[0])
    #     idx1 = list(np.where(choices0 == 1)[0]) + list(np.where(choices1 == 2)[0])
    #     idx2 = list(np.where(choices1 == 1)[0]) + list(np.where(choices2 == 2)[0])
    #     idx3 = list(np.where(choices1 == 0)[0]) + list(np.where(choices2 == 3)[0])
    #     idx4 = list(np.where(choices0 == 3)[0]) + list(np.where(choices2 == 0)[0])
    #     idx5 = list(np.where(choices0 == 2)[0]) + list(np.where(choices2 == 1)[0])
    #
    #     # idx0, idx1, etc. represent the indexes of the different type of absolute exchange :
    #     # 0 : 0 -> 1
    #     # 1 : 0 -> 2
    #     # 2 : 1 -> 0
    #     # 3 : 1 -> 2
    #     # 4 : 2 -> 0
    #     # 5 : 2 -> 1
    #
    #     idx_type_of_good = [idx0, idx1, idx2, idx3, idx4, idx5]
    #
    #     for i in range(6):
    #
    #         if np.asarray(idx_type_of_good[i]).size:
    #             self.results[self.t, i] = np.mean(self.finding_a_partner[idx_type_of_good[i]])
    #
    #         else:
    #             self.results[self.t, i] = -1.
    #
    #
    # cdef compute_noise(self, cnp.ndarray array):
    #
    #     cdef:
    #         float mean, noise
    #
    #     if array.size:
    #         mean = np.mean(array)
    #         if mean != 0 and mean != 1:
    #             noise = -np.log2(1 - mean) + mean * np.log2((1 - mean) / mean)
    #         else:
    #             noise = 0
    #     else:
    #         noise = -1.
    #
    #     return noise
    #
    # cdef compute_distortion(self, float personal_info_mean, float total_info_mean):
    #
    #     cdef:
    #         float distortion
    #
    #     if personal_info_mean != -1. and total_info_mean != -1.:
    #         distortion = personal_info_mean - total_info_mean
    #     else:
    #         distortion = -1.
    #     return distortion

    cdef info_choice_transition_function(self, x, y):

        return self.choice_transition[x, y]

    cdef update_estimations(self):

        # a = time()

        cdef:
            list agents_list
            int i_type, i_choice, i, k, j
            cnp.ndarray agents, informers, agents_types, agents_results, agents_choices, relative_agents_type,\
                relative_total_types, relative_choices, relative_total_choices, averages, total_mean, id0, id1,\
                my_opinion, others_opinion, estimation_types, noise, averages_for_agent0, averages_for_agent1,\
                averages_for_agent2

        # distortion_ij = np.zeros(self.n)
        # distortion_ik = np.zeros(self.n)
        # distortion_kj = np.zeros(self.n)
        # distortion_ki = np.zeros(self.n)
        #
        # noise_ij = np.zeros(self.n)
        # noise_ik = np.zeros(self.n)
        # noise_kj = np.zeros(self.n)
        # noise_ki = np.zeros(self.n)
        #
        # averages_for_agent0 = np.zeros([4, len(self.idx0)])     # mean of type agent 0 informers results
        # averages_for_agent1 = np.zeros([4, len(self.idx1)])     # mean of type agent 1 informers results
        # averages_for_agent2 = np.zeros([4, len(self.idx2)])     # mean of type agent 2 informers results

        for i in range(self.n):

            informers_choices, informers_results = self.info_select_informers(i)

            # averages and total_mean are respectively the means of the
            informers_averages = self.info_compute_information(
                i=i,
                informers_choices=informers_choices,
                informers_results=informers_results)

            estimations = \
                np.array([self.estimation_ij[i], self.estimation_ik[i], self.estimation_kj[i], self.estimation_ki[i]])

            new_estimations = \
                self.info_integrate_information(i=i, informers_averages=informers_averages, estimations=estimations)



            self.estimation_ij[i] = new_estimations[0]
            self.estimation_ik[i] = new_estimations[1]
            self.estimation_kj[i] = new_estimations[2]
            self.estimation_ki[i] = new_estimations[3]

        # b = time()
        # self.dico['update_estimations'][0] += (b-a)
        # self.dico['update_estimations'][1] += 1

    cdef info_select_informers(self, i):

        # i is idx of agent for who we select informers

        i_type = self.type[i]
        i_choice = self.i_choice[i]

        # Here, we take the type and the choice of agent i, in order to compare him with the other agents
        # and to compute his new estimation on the easiness to make the transaction he chose.
        # Throughout the rest, each array or list containing 4 elements will correspond to estimation_ij,
        # estimation_ik, estimation_kj and estimation_ki in this order.

        agents = np.arange(self.n)
        agents_list = list(agents)
        agents_list.remove(i)

        informers = np.random.choice(agents_list, self.q_information, replace=False)
        # We choose here a certain number (which corresponds to the quantity of information q_information)
        # of informers among all the agents

        informers_types = self.type[informers]
        informers_choices = self.i_choice[informers]
        informers_results = self.finding_a_partner[informers]

        relative_type = (informers_types - i_type) % 3

        # Here, we compute the relative choices for each agent i made by his informers, depending on their types and
        # their choice. We also compute all the relative choices i made by all the agents relatively to each aent i,
        # in order to compute the distortion.

        relative_choices = self.info_choice_transition_function(relative_type, informers_choices)

        return relative_choices, informers_results

    cdef info_compute_information(self, i, informers_choices, informers_results):

        averages = np.zeros(4)
        # noise = np.zeros(4)

        # print(relative_choices)

        for exchange_type in range(4):

            # Get (relative) id of informers that bring information for this particular type of exchange

            bool_exchange = informers_choices[:] == exchange_type

            if np.sum(bool_exchange) > 0.:

                averages[exchange_type] = np.mean(informers_results[bool_exchange])
            else:
                averages[exchange_type] = -1.

        return averages

            # if i in self.idx0:
            #     averages_for_agent0[j, i] = averages[j]
            # if i in self.idx1:
            #     k = i - self.idx1[0]
            #     averages_for_agent1[j, k] = averages[j]
            # if i in self.idx2:
            #     k = i - self.idx1[0]
            #     averages_for_agent2[j, k] = averages[j]
            #
            # if id1.size:
            #     total_mean[j] = np.mean(self.finding_a_partner[id1])
            # else:
            #     total_mean[j] = -1.
            #
            # noise[j] = self.compute_noise(agents_results[id0])
            #
            # self.total_received_information[i] += len(id0)

    cdef info_integrate_information(self, i, informers_averages, estimations):

        i_choice = self.i_choice[i]

        # Here, we have computed the right type of transactions from the point of view of agent i compared with
        # the type of choices which are made by the other agents. Then, once we have identified what agents
        # contributes to the 4 different estimations for i, we give their results corresponding to their
        # previous transaction according to the fact they succeeded in their transaction or not.

        my_opinion = np.zeros(4)
        my_opinion[i_choice] = self.epsilon * (self.finding_a_partner[i] - estimations[i_choice])

        for exchange_type in range(4):

            if informers_averages[exchange_type] != -1.:  # Exclude this value ; assume others_opinion[k] is null

                others_opinion = \
                    (1 - self.epsilon) * (informers_averages[exchange_type] - estimations[exchange_type])

            else:

                others_opinion = 0

            estimations[exchange_type] += self.alpha * (my_opinion[exchange_type] + others_opinion)

        for exchange_type in range(4):

            estimations[exchange_type] += self.alpha * (my_opinion[exchange_type] + others_opinion)

        return estimations

        # distortion_ij[i] = self.compute_distortion(averages[0], total_mean[0])
            # distortion_ik[i] = self.compute_distortion(averages[1], total_mean[1])
            # distortion_kj[i] = self.compute_distortion(averages[2], total_mean[2])
            # distortion_ki[i] = self.compute_distortion(averages[3], total_mean[3])
            #
            # noise_ij[i] = noise[0]
            # noise_ik[i] = noise[1]
            # noise_kj[i] = noise[2]
            # noise_ki[i] = noise[3]



        # for k in range(4):
        #     a0 = averages_for_agent0[k, :]
        #     a1 = a0[a0[:]!=-1.]
        #     b0 = averages_for_agent1[k, :]
        #     b1 = b0[b0[:]!=-1.]
        #     c0 = averages_for_agent2[k, :]
        #     c1 = c0[c0[:]!=-1.]
        #     if a1.size:
        #         self.informers_results_mean_for_agent0[self.t, k] = np.mean(a1)
        #     else:
        #         self.informers_results_mean_for_agent0[self.t, k] = -1.
        #     if b1.size:
        #         self.informers_results_mean_for_agent1[self.t, k] = np.mean(b1)
        #     else:
        #         self.informers_results_mean_for_agent1[self.t, k] = -1.
        #     if c1.size:
        #         self.informers_results_mean_for_agent2[self.t, k] = np.mean(c1)
        #     else:
        #         self.informers_results_mean_for_agent2[self.t, k] = -1.
        #
        # for j in range(3):
        #     self.distortion_ij[self.t, j] = self.compute_average_and_error(distortion_ij)[0][j]
        #     self.distortion_ik[self.t, j] = self.compute_average_and_error(distortion_ik)[0][j]
        #     self.distortion_kj[self.t, j] = self.compute_average_and_error(distortion_kj)[0][j]
        #     self.distortion_ki[self.t, j] = self.compute_average_and_error(distortion_ki)[0][j]
        #     self.noise_ij[self.t, j] = self.compute_average_and_error(noise_ij)[0][j]
        #     self.noise_ik[self.t, j] = self.compute_average_and_error(noise_ik)[0][j]
        #     self.noise_kj[self.t, j] = self.compute_average_and_error(noise_kj)[0][j]
        #     self.noise_ki[self.t, j] = self.compute_average_and_error(noise_ki)[0][j]
        #     self.distortion_ij_error[self.t, j] = self.compute_average_and_error(distortion_ij)[1][j]
        #     self.distortion_ik_error[self.t, j] = self.compute_average_and_error(distortion_ik)[1][j]
        #     self.distortion_kj_error[self.t, j] = self.compute_average_and_error(distortion_kj)[1][j]
        #     self.distortion_ki_error[self.t, j] = self.compute_average_and_error(distortion_ki)[1][j]
        #     self.noise_ij_error[self.t, j] = self.compute_average_and_error(noise_ij)[1][j]
        #     self.noise_ik_error[self.t, j] = self.compute_average_and_error(noise_ik)[1][j]
        #     self.noise_kj_error[self.t, j] = self.compute_average_and_error(noise_kj)[1][j]
        #     self.noise_ki_error[self.t, j] = self.compute_average_and_error(noise_ki)[1][j]

    # cdef compute_average_and_error(self, cnp.ndarray array):
    #
    #     cdef:
    #         cnp.ndarray a0, a1, a2
    #         float m0, m1, m2, e0, e1, e2
    #
    #     a0 = array[self.idx0]
    #     a1 = array[self.idx1]
    #     a2 = array[self.idx2]
    #
    #     if a0[a0[:] != -1.].size:
    #         m0 = np.mean(a0[a0[:] != -1.])
    #         e0 = np.std(a0[a0[:] != -1.])
    #     else:
    #         m0 = -0.001
    #         e0 = 0
    #
    #     if a1[a1[:] != -1.].size:
    #         m1 = np.mean(a1[a1[:] != -1.])
    #         e1 = np.std(a1[a1[:] != -1.])
    #     else:
    #         m1 = -0.001
    #         e1 = 0
    #
    #     if a2[a2[:] != -1.].size:
    #         m2 = np.mean(a2[a2[:] != -1.])
    #         e2 = np.std(a2[a2[:] != -1.])
    #     else:
    #         m2 = -0.001
    #         e2 = 0
    #
    #     return [[m0, m1, m2], [e0, e1, e2]]

    cdef test_for_money(self):

        # a = time()

        cdef:
            int money, cond0, cond1, cond2, cond3, cond4, cond5

        money = -1

        # Money = 0?
        # type '0' should use direct exchange
        cond0 = np.mean(self.i_choice[self.idx0] == 0) > self.money_threshold

        # type '1' should use indirect exchange
        cond1 = np.mean([i in [1, 2] for i in self.i_choice[self.idx1]]) > self.money_threshold

        # type '2' should use direct exchange
        cond2 = np.mean(self.i_choice[self.idx2] == 0) > self.money_threshold

        if (cond0 * cond1 * cond2) == 1:

            money = 0

        else:

            # Money = 1?
            cond0 = np.mean(self.i_choice[self.idx0] == 0) > self.money_threshold
            cond1 = np.mean(self.i_choice[self.idx1] == 0) > self.money_threshold
            cond2 = np.mean([i in [1, 2] for i in self.i_choice[self.idx2]]) > self.money_threshold

            if (cond0 * cond1 * cond2) == 1:

                money = 1

            else:

                # Money = 2?
                cond0 = np.mean([i in [1, 2] for i in self.i_choice[self.idx0]]) > self.money_threshold
                cond1 = np.mean(self.i_choice[self.idx1] == 0) > self.money_threshold
                cond2 = np.mean(self.i_choice[self.idx2] == 0) > self.money_threshold

                if (cond0 * cond1 * cond2) == 1:

                    money = 2

        ##### END TEST #####
        ##### SAVE RESULT ####

        if money == -1:

            self.money_round[:] = 0
            self.money_time[:] = 0

        else:
            self.money_round[(money+1) % 3] = 0
            self.money_round[(money+2) % 3] = 0
            self.money_round[money] += 1
            # Memorize time for money emergence
            if self.money_round[money] == 1:
                self.money_time[money] = self.t

        # b = time()
        # self.dico['test_for_money'][0] += (b-a)
        # self.dico['test_for_money'][1] += 1

    cpdef determine_money(self):

        # a = time()

        cdef:
            int i

        for i in range(3):

            if self.money_round[i] >= self.money_delta:

                self.money = i
                self.t_emergence = self.money_time[i]
                break

        # b = time()
        # self.dico['determine_money'][0] += (b-a)
        # self.dico['determine_money'][1] += 1

    cpdef compute_welfare(self, moment):

        # a = time()

        cdef:
            cnp.ndarray d_exchange, i_exchange

        # Compute rewards number
        d_exchange = np.where((self.i_choice == 0) * (self.finding_a_partner == 1))[0]
        i_exchange = np.where((self.i_choice == 2) * (self.finding_a_partner == 1))[0]

        if moment == 0:
            self.beginning_reward[self.t_beginning_reward, d_exchange] += 1
            self.beginning_reward[self.t_beginning_reward, i_exchange] += 1
            self.t_beginning_reward += 1

        else:

            self.end_reward[self.t_end_reward, d_exchange] += 1
            self.end_reward[self.t_end_reward, i_exchange] += 1
            self.t_end_reward += 1

        # b = time()
        # self.dico['compute_welfare'][0] += (b-a)
        # self.dico['compute_welfare'][1] += 1

    cpdef summarise_welfare(self, moment):

        # a0 = time()

        if moment == 0:

            reward = self.beginning_reward
            welfare = self.beginning_welfare

        else:
            reward = self.end_reward
            welfare = self.end_welfare

        a = np.zeros((self.welfare_delta, len(self.workforce)))

        for i in range(self.welfare_delta):

            a[i, 0] = np.mean(reward[i, self.idx0])
            a[i, 1] = np.mean(reward[i, self.idx1])
            a[i, 2] = np.mean(reward[i, self.idx2])

        welfare[0] = np.mean(a[:, 0])
        welfare[1] = np.mean(a[:, 1])
        welfare[2] = np.mean(a[:, 2])

        # b0 = time()
        # self.dico['summarize_welfare'][0] += (b0-a0)
        # self.dico['summarize_welfare'][1] += 1


class SimulationRunner(object):

    def __init__(self, parameters):

        # Time the simulation should last
        self.time_limit = parameters["time_limit"]

        self.welfare_delta = parameters["welfare_delta"]

        # Create the economy to simulate
        self.eco = Economy(parameters)

    def run(self):

        assert self.time_limit > self.welfare_delta*2, "Time limit can't be inferior to double of welfare delta!"

        self.eco.setup()

        # Run simulation for as time units as required.
        for t in range(0, self.welfare_delta):

            self.eco.run()
            self.eco.compute_welfare(0)

        for t in range(self.welfare_delta, self.time_limit-self.welfare_delta):

            self.eco.run()

        for t in range(self.time_limit-self.welfare_delta, self.time_limit):

            self.eco.run()
            self.eco.compute_welfare(1)

        self.eco.determine_money()
        self.eco.summarise_welfare(0)
        self.eco.summarise_welfare(1)

        return self.export()

    def export(self):

        # distortion_folder = 'results/Distortion'
        # noise_folder = 'results/Noise'
        # choice_freq_folder = 'results/Choice_Freq'
        # informers_results_folder = 'results/Informers_Results_By_Agent_Type'
        # total_results_folder = 'results/Total_Results'
        #
        # np.savetxt('{}/Distortion_ij.txt'.format(distortion_folder), self.eco.distortion_ij, delimiter=' ')
        # np.savetxt('{}/Distortion_ik.txt'.format(distortion_folder), self.eco.distortion_ik, delimiter=' ')
        # np.savetxt('{}/Distortion_kj.txt'.format(distortion_folder), self.eco.distortion_kj, delimiter=' ')
        # np.savetxt('{}/Distortion_ki.txt'.format(distortion_folder), self.eco.distortion_ki, delimiter=' ')
        #
        # np.savetxt('{}/Distortion_ij_error.txt'.format(distortion_folder), self.eco.distortion_ij_error, delimiter=' ')
        # np.savetxt('{}/Distortion_ik_error.txt'.format(distortion_folder), self.eco.distortion_ik_error, delimiter=' ')
        # np.savetxt('{}/Distortion_kj_error.txt'.format(distortion_folder), self.eco.distortion_kj_error, delimiter=' ')
        # np.savetxt('{}/Distortion_ki_error.txt'.format(distortion_folder), self.eco.distortion_ki_error, delimiter=' ')
        #
        # np.savetxt('{}/Noise_ij.txt'.format(noise_folder), self.eco.noise_ij, delimiter=' ')
        # np.savetxt('{}/Noise_ik.txt'.format(noise_folder), self.eco.noise_ik, delimiter=' ')
        # np.savetxt('{}/Noise_kj.txt'.format(noise_folder), self.eco.noise_kj, delimiter=' ')
        # np.savetxt('{}/Noise_ki.txt'.format(noise_folder), self.eco.noise_ki, delimiter=' ')
        #
        # np.savetxt('{}/Noise_ij_error.txt'.format(noise_folder), self.eco.noise_ij_error, delimiter=' ')
        # np.savetxt('{}/Noise_ik_error.txt'.format(noise_folder), self.eco.noise_ik_error, delimiter=' ')
        # np.savetxt('{}/Noise_kj_error.txt'.format(noise_folder), self.eco.noise_kj_error, delimiter=' ')
        # np.savetxt('{}/Noise_ki_error.txt'.format(noise_folder), self.eco.noise_ki_error, delimiter=' ')
        #
        # np.savetxt('{}/choice_agent0.txt'.format(choice_freq_folder), self.eco.choice_agent0, delimiter=' ')
        # np.savetxt('{}/choice_agent1.txt'.format(choice_freq_folder), self.eco.choice_agent1, delimiter=' ')
        # np.savetxt('{}/choice_agent2.txt'.format(choice_freq_folder), self.eco.choice_agent2, delimiter=' ')
        #
        # np.savetxt('{}/results_informers_type0.txt'.format(informers_results_folder),
        #            self.eco.informers_results_mean_for_agent0, delimiter=' ')
        # np.savetxt('{}/results_informers_type1.txt'.format(informers_results_folder),
        #            self.eco.informers_results_mean_for_agent1, delimiter=' ')
        # np.savetxt('{}/results_informers_type2.txt'.format(informers_results_folder),
        #            self.eco.informers_results_mean_for_agent2, delimiter=' ')
        #
        # np.savetxt('{}/total_results.txt'.format(total_results_folder),
        #            self.eco.results, delimiter=' ')

        data = {"x0": self.eco.workforce[0],
                "x1": self.eco.workforce[1],
                "x2": self.eco.workforce[2],
                "alpha": self.eco.alpha,
                "tau": self.eco.temperature,
                "money": self.eco.money,
                "t_emergence": self.eco.t_emergence,
                "beginning_welfare_x0": self.eco.beginning_welfare[0],
                "beginning_welfare_x1": self.eco.beginning_welfare[1],
                "beginning_welfare_x2": self.eco.beginning_welfare[2],
                "end_welfare_x0": self.eco.end_welfare[0],
                "end_welfare_x1": self.eco.end_welfare[1],
                "end_welfare_x2": self.eco.end_welfare[2],
                "useful_information_mean": (np.mean(self.eco.total_received_information)/self.time_limit)/self.eco.n,
                "useful_information_standard_deviation": np.std(self.eco.total_received_information/self.time_limit),
                "useful_information_maximum": np.max(self.eco.total_received_information/self.time_limit),
                "useful_information_minimum": np.min(self.eco.total_received_information/self.time_limit),
                "information_by_round": np.round(self.eco.q_information/self.eco.n, 1),
                "epsilon": self.eco.epsilon}

        np.save('data.npy', data)

        return data