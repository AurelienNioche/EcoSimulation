from pickle import dump
from os import path, mkdir


class BackUp(object):

    @classmethod
    def save_data(cls, results, parameters, root_folder="../data"):

        cls.create_folders(root_folder)

        print("\nSaving data...")

        saving_name = "{date}_idx{idx}".format(date=parameters["date"], idx=parameters["idx"])

        direct_exchanges = results["direct_exchanges"]
        indirect_exchanges = results["indirect_exchanges"]

        dump(direct_exchanges,
             open("{}/exchanges/direct_exchanges_{}.p".format(root_folder, saving_name), mode='wb'))

        dump(indirect_exchanges,
             open("{}/exchanges/indirect_exchanges_{}.p".format(root_folder, saving_name), mode='wb'))

        dump(parameters,
             open("{}/parameters/parameters_{}.p".format(root_folder, saving_name), mode='wb'))

        print("\nData saved.")

    @classmethod
    def create_folders(cls, root_folder):

        folders = [root_folder, "{}/parameters".format(root_folder), "{}/exchanges".format(root_folder)]
        for i in folders:

            if not path.exists(i):

                mkdir(i)
