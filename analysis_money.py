import numpy as np
from multiprocessing import Pool
from module.data_importer import DataImporter
from module.save_db_dic import BackUp
from collections import OrderedDict
from os import path


# ------------------------------------------------||| MONEY TEST |||----------------------------------------------- #

class MoneyAnalysis(object):

    def __init__(self, data_folder):

        self.money_threshold = .90
        self.data_folder = data_folder

    def test_for_money_state(self, direct_exchange, indirect_exchange):

        money = -1

        # Money = 0?
        # type '0' should use direct exchange
        cond0 = direct_exchange[0] > self.money_threshold

        # type '1' should use indirect exchange
        cond1 = indirect_exchange[1] > self.money_threshold

        # type '2' should use direct exchange
        cond2 = direct_exchange[2] > self.money_threshold

        if (cond0 * cond1 * cond2) == 1:

            money = 0

        else:

            # Money = 1?
            cond0 = direct_exchange[0] > self.money_threshold
            cond1 = direct_exchange[1] > self.money_threshold
            cond2 = indirect_exchange[2] > self.money_threshold

            if (cond0 * cond1 * cond2) == 1:

                money = 1

            else:

                # Money = 2?
                cond0 = indirect_exchange[0] > self.money_threshold
                cond1 = direct_exchange[1] > self.money_threshold
                cond2 = direct_exchange[2] > self.money_threshold

                if (cond0 * cond1 * cond2) == 1:
                    money = 2

        return money

    def analyse(self, suffix):

        # Import data
        data_importer = DataImporter(self.data_folder)
        data = data_importer.import_data_for_single_session(suffix)

        parameters = data["parameters"]
        direct_exchange = data["direct_exchanges"]
        indirect_exchange = data["indirect_exchanges"]

        money_timeline = np.zeros(parameters["t_max"])
        money = {0: 0, 1: 0, 2: 0, -1: 0}
        interruptions = 0

        for t in range(parameters["t_max"]):

            money_t = self.test_for_money_state(direct_exchange=direct_exchange[t],
                                                indirect_exchange=indirect_exchange[t])
            money_timeline[t] = money_t
            money[money_t] += 1

            if t > 0:
                cond0 = money_t == -1
                cond1 = money_timeline[t - 1] != -1
                interruptions += cond0 * cond1

        data_to_save = OrderedDict([
            ('idx', parameters["idx"]),
            ('a0', parameters["workforce"][0]),
            ('a1', parameters["workforce"][1]),
            ('a2', parameters["workforce"][2]),
            ('alpha', parameters["alpha"]),
            ('tau', parameters["tau"]),
            ('t_max', parameters["t_max"]),
            ('epsilon', parameters["epsilon"]),
            ('q_information', parameters["q_information"]),
            ('m0', money[0]),
            ('m1', money[1]),
            ('m2', money[2]),
            ('m_sum', money[0] + money[1] + money[2]),
            ('interruptions', interruptions)
        ])

        return data_to_save


# ------------------------------------------------||| Data Saver |||----------------------------------------------- #

class DataSaver(object):
    def __init__(self, data_folder, result_folder):

        self.data_folder = data_folder
        self.result_folder = result_folder

    def save_data(self):

        data_importer = DataImporter(data_folder=self.data_folder)
        suffixes = data_importer.import_suffix_list()

        money_analysis = MoneyAnalysis(self.data_folder)

        pool = Pool(processes=8)
        data = pool.map(money_analysis.analyse, suffixes)

        self.write(data)

    def write(self, data):
        backup = BackUp(results_folder=self.result_folder, database_name="results")
        backup.save(data=data)


# ------------------------------------------------||| MAIN  |||----------------------------------------------- #


def main():

    data_folder = "/Users/M-E4-ANIOCHE/Desktop/data"
    assert path.exists(data_folder), "Wrong path to data..."
    result_folder = "/Users/M-E4-ANIOCHE/Desktop/results"

    data_saver = DataSaver(data_folder=data_folder, result_folder=result_folder)
    data_saver.save_data()


if __name__ == "__main__":

    main()
