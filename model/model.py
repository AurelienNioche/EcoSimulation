import numpy as np


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


class Economy:

    def __init__(self, x0, x1, x2, t_max, alpha, tau, q, gamma):

        # ------------- #
        # Parameters

        # np.random.seed(parameters["seed"])
        # ------------ #

        self.n = x0 + x1 + x2  # Total number of agents
        self.workforce = np.zeros(3, dtype=int)
        self.workforce[:] = x0, x1, x2  # Number of agents by type

        self.t_max = t_max

        self.alpha = alpha  # Learning coefficient
        self.temperature = tau  # Softmax parameter

        self.q_information = int(np.round(q*self.n))

        if self.q_information == self.n:
            self.q_information -= 1

        self.gamma = gamma  # Parameter for the weight of own information
        # against information provided by others

        # --------- #
        # --------- #

        # To convert relative choice for agent a for an agent b
        # knowing that type of agent b is 0, +1, +2 relatively to agent b
        self.choice_transition = np.array([
            [0, 1, 2, 3],
            [-1, -1, 1, 0],
            [3, 2, -1, -1]], dtype=int)

        self.type = np.zeros(self.n, dtype=int)

        self.type[:] = np.concatenate(([0, ]*self.workforce[0],
                                       [1, ]*self.workforce[1],
                                       [2, ]*self.workforce[2]))

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

        # self.received_information = np.zeros((self.n, 4))
        # self.total_received_information = np.zeros(self.n, dtype=int)

    def setup(self):

        self.estimation_ij[:] = np.random.random(self.n)
        self.estimation_ik[:] = np.random.random(self.n)
        self.estimation_kj[:] = np.random.random(self.n)
        self.estimation_ki[:] = np.random.random(self.n)

    def run(self):

        # Make agents updating the decision they are facing
        self.update_decision()

        # Make agents updating the values they attribute to options
        self.update_options_values()

        # Make agents choosing
        self.make_a_choice()

        # Move the agents where they are supposed to go
        self.who_is_where()

        # Realize the transactions in the different markets
        self.make_the_transactions()

        # Make agents learn about the success rates of each type of exchange knowing the results of other agents
        # and themselves.
        self.update_estimations()


############################################################
#            CHOICE                                        #
############################################################

    def update_decision(self):

        # a = time()

        # Set the decision each agent faces at time t, according to the fact he succeeded or not in his exchange at t-1,
        #  the decision he previously faced, and the choice he previously made.
        self.decision[:] = self.decision_array[self.finding_a_partner,
                                               self.decision,
                                               self.choice]
        # b = time()
        # self.dico['update_decision'][0] += (b-a)
        # self.dico['update_decision'][1] += 1

    def update_options_values(self):

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

    def make_a_choice(self):

        # a = time()

        id0 = np.where(self.decision == 0)[0]
        id1 = np.where(self.decision == 1)[0]

        self.value_option0[id0] = self.value_ij[id0]
        self.value_option1[id0] = self.value_ik[id0]

        self.value_option0[id1] = self.value_kj[id1]
        self.value_option1[id1] = self.value_ki[id1]

        # Set a probability to current option 0 using softmax rule
        # (As there is only 2 options each time, computing probability for a unique option is sufficient)

        self.probability_of_choosing_option0[:] = \
            1 / (1 + np.exp(-(self.value_option0 - self.value_option1)/self.temperature))

        self.random_number[:] = np.random.uniform(0., 1., self.n)  # Generate random numbers

        # Make a choice using the probability of choosing option 0 and a random number for each agent
        # Choose option 1 if random number > or = to probability of choosing option 0,
        #  choose option 0 otherwise
        self.choice[:] = self.random_number >= self.probability_of_choosing_option0
        self.i_choice[:] = (self.decision * 2) + self.choice

    def who_is_where(self):

        # Place the agents according to their type, decision and choice
        self.place[:] = self.placement[self.type, self.decision, self.choice]

    def make_the_transactions(self):

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


############################################################
#            INFORMATION                                   #
############################################################


    def info_choice_transition_function(self, x, y):

        return self.choice_transition[x, y]

    def update_estimations(self):

        # a = time()

        for i in range(self.n):

            informers_choices, informers_results = self.info_select_informers(i)

            informers_averages = self.info_compute_information(
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

    def info_select_informers(self, i):

        # i is idx of agent for who we select informers

        i_type = self.type[i]

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

    def info_compute_information(self, informers_choices, informers_results):

        averages = np.zeros(4)

        for exchange_type in range(4):

            # Get (relative) id of informers that bring information for this particular type of exchange

            bool_exchange = informers_choices[:] == exchange_type

            if np.sum(bool_exchange) > 0:

                averages[exchange_type] = np.mean(informers_results[bool_exchange])
            else:
                averages[exchange_type] = np.nan

        return averages

    def info_integrate_information(self, i, informers_averages, estimations):

        i_choice = self.i_choice[i]

        # Here, we have computed the right type of transactions from the point of view of agent i compared with
        # the type of choices which are made by the other agents. Then, once we have identified what agents
        # contributes to the 4 different estimations for i, we give their results corresponding to their
        # previous transaction according to the fact that they succeeded in their transaction or not.

        my_opinion = np.zeros(4)
        my_opinion[i_choice] = self.gamma * (self.finding_a_partner[i] - estimations[i_choice])

        for exchange_type in range(4):

            if not np.isnan(informers_averages[exchange_type]):  # Exclude this value ; assume others_opinion[k] is null

                others_opinion = \
                    (1 - self.gamma) * (informers_averages[exchange_type] - estimations[exchange_type])

            else:
                others_opinion = 0

            estimations[exchange_type] += self.alpha * (my_opinion[exchange_type] + others_opinion)

        return estimations
