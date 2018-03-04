from os import path, mkdir
from sqlite3 import connect, OperationalError
import numpy as np
from datetime import date
from ___old__.module import Folders


class BackUp(object):

    @classmethod
    def save_data(cls, results, parameters, backup_file):

        root_folder = Folders.data

        if not path.exists(root_folder):

            mkdir(root_folder)

        print("\nSaving data...")

        saving_name = "{date}_idx{idx}".format(date=parameters["date"], idx=parameters["eco_idx"])

        queries = list()
        queries.append("CREATE TABLE `direct_exchanges_{}` (ID INT PRIMARY KEY, `x0` REAL, `x1` REAL, `x2` REAL)"
                       .format(saving_name))

        queries.append("CREATE TABLE `indirect_exchanges_{}` (ID INT PRIMARY KEY, `x0` REAL, `x1` REAL, `x2` REAL)"
                       .format(saving_name))

        queries.append("CREATE TABLE IF NOT EXISTS `parameters` (ID INT PRIMARY KEY, `x0` INT, `x1` INT, `x2` INT, "
                       "`t_max` INT, `alpha` REAL, `tau` REAL, `gamma` REAL, `q` REAL, `eco_idx` INT, "
                       "`date` TEXT)")

        connection = connect("{}/{}.db".format(root_folder, backup_file))
        cursor = connection.cursor()

        for query in queries:
            try:
                cursor.execute(query)
            except OperationalError as e:
                print(query)
                raise e

        n = len(results["direct_exchanges"])
        direct_exchanges = np.concatenate((np.arange(n).reshape(n, 1), results["direct_exchanges"]), axis=1)
        indirect_exchanges = np.concatenate((np.arange(n).reshape(n, 1), results["indirect_exchanges"]), axis=1)

        cursor.executemany("INSERT INTO `direct_exchanges_{}` (ID, x0, x1, x2) VALUES (?, ?, ?, ?)".format(saving_name),
                           direct_exchanges)
        cursor.executemany("INSERT INTO `indirect_exchanges_{}` (ID, x0, x1, x2) VALUES (?, ?, ?, ?)".format(saving_name),
                           indirect_exchanges)

        cursor.execute("SELECT max(ID) FROM `parameters`")
        max_id = cursor.fetchone()[0]
        if max_id is None:
            ID = 0
        else:
            ID = max_id + 1

        query = \
            "INSERT INTO `parameters` (ID, `x0`, `x1`, `x2`, " \
            "`t_max`, `alpha`, `tau`, `gamma`, `q`, `eco_idx`, `date`) " \
            "VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, '{}')".format(
                ID,
                parameters["x0"], parameters["x1"], parameters["x2"],
                parameters["t_max"], parameters["alpha"], parameters["tau"],
                parameters["gamma"], parameters["q"], parameters["eco_idx"], parameters["date"])
        try:
            cursor.execute(query)
        except OperationalError as e:
            print(query)
            raise e

        print("\nData saved.")
        connection.commit()
        connection.close()


def main():

    n = 1000

    results = {
        "direct_exchanges": np.random.random((n, 3)),
        "indirect_exchanges": np.random.random((n, 3))
    }
    parameters = {
        "x0": np.random.randint(500),
        "x1": np.random.randint(500),
        "x2": np.random.randint(500),
        "t_max": np.random.randint(500),
        "alpha": np.random.random(),
        "tau": np.random.random(),
        "gamma": np.random.random(),
        "q": np.random.random(),
        "eco_idx": np.random.randint(500),
        "date":  str(date.today())
    }

    BackUp.save_data(results=results, parameters=parameters, backup_file="backup_file")


if __name__ == '__main__':

    main()
