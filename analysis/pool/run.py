from pylab import np, plt


class MoneyAnalysis(object):

    money_threshold = .75

    @classmethod
    def _test_for_money_state(cls, direct_exchange, indirect_exchange):

        money = -1

        # Money = 0?
        # type '0' should use direct exchange
        cond0 = direct_exchange[0] > cls.money_threshold

        # type '1' should use indirect exchange
        cond1 = indirect_exchange[1] > cls.money_threshold

        # type '2' should use direct exchange
        cond2 = direct_exchange[2] > cls.money_threshold

        if (cond0 * cond1 * cond2) == 1:

            money = 0

        else:

            # Money = 1?
            cond0 = direct_exchange[0] > cls.money_threshold
            cond1 = direct_exchange[1] > cls.money_threshold
            cond2 = indirect_exchange[2] > cls.money_threshold

            if (cond0 * cond1 * cond2) == 1:

                money = 1

            else:

                # Money = 2?
                cond0 = indirect_exchange[0] > cls.money_threshold
                cond1 = direct_exchange[1] > cls.money_threshold
                cond2 = direct_exchange[2] > cls.money_threshold

                if (cond0 * cond1 * cond2) == 1:
                    money = 2

        return money

    @classmethod
    def run(cls, results):

        direct_exchange, indirect_exchange = results.direct_exchanges, results.indirect_exchanges

        t_max = len(direct_exchange)

        money_time_line = np.zeros(t_max)
        money = {0: 0, 1: 0, 2: 0, -1: 0}
        interruptions = 0

        for t in range(t_max):

            money_t = cls._test_for_money_state(
                direct_exchange=direct_exchange[t],
                indirect_exchange=indirect_exchange[t])
            money_time_line[t] = money_t
            money[money_t] += 1

            if t > 0:

                cond0 = money_t == -1
                cond1 = money_time_line[t-1] != -1
                interruptions += cond0 * cond1

        # return {"money": money, "interruptions": interruptions}
        return max([money[0], money[1], money[2]])


def run(pool_results):

    # r = pool_results.results[30]
    # plt.plot(r.indirect_exchanges)
    # plt.show()
    # m = MoneyAnalysis.run(r)
    # print(m)
    #
    # for i, r in enumerate(pool_results.results):
    #
    #     plt.plot(r.indirect_exchanges)
    #     plt.title(i)
    #     plt.show()

    alphas = []
    taus = []
    qs = []
    gammas = []

    y = []

    for i, r in enumerate(pool_results.results):

        # if not r.parameters["x0"] == r.parameters["x1"] == r.parameters["x2"]:
        #    continue

        alphas.append(r.parameters["alpha"])
        taus.append(r.parameters["tau"])
        qs.append(r.parameters["q"])
        gammas.append(r.parameters["gamma"])
        m = MoneyAnalysis.run(r)
        print(i, m)
        y.append(m)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(221)
    ax.scatter(taus, y, c="black", alpha=0.4, s=15)
    ax.set_ylabel("n monetary states")
    ax.set_xlabel(r"$\tau$")

    ax = fig.add_subplot(222)
    ax.scatter(alphas, y, c="black", alpha=0.4, s=15)
    ax.set_ylabel("n monetary states")
    ax.set_xlabel(r"$\alpha$")

    ax = fig.add_subplot(223)
    ax.scatter(qs, y, c="black", alpha=0.4, s=15)
    ax.set_ylabel("n monetary states")
    ax.set_xlabel(r"$q$")

    ax = fig.add_subplot(224)
    ax.scatter(gammas, y, c="black", alpha=0.4, s=15)
    ax.set_ylabel("n monetary states")
    ax.set_xlabel(r"$\gamma$")

    # ax = fig.add_subplot(224)
    # ax.scatter(x[X.x], y, c="black", alpha=0.4, s=15)
    # ax.set_ylabel("n monetary states")
    # ax.set_xlabel(r"n agents")

    # plt.text(0.005, 0.005, results_pool.file_name, transform=fig.transFigure, fontsize='x-small', color='0.5')

    plt.tight_layout()

    # plt.savefig("{}/separate_indirect_exchanges_proportions_{}.pdf"
    #           .format(analysis.parameters.fig_folder, results_pool.file_name))

    plt.show()

    #plt.scatter(alphas, y, color='black', alpha=0.5)
    #plt
    # plt.show()

