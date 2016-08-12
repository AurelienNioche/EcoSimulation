import pickle
from os import listdir
from os.path import isfile, join, exists


class DataImporter(object):

    def __init__(self, data_folder):

        assert exists(data_folder), "Wrong name for data folder..."

        self.folder = {
            "parameters": "{}/parameters".format(data_folder),
            "exchanges": "{}/exchanges".format(data_folder)
        }

    def import_suffix_list(self):

        mypath = self.folder["parameters"]
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        suffixes = [f.split("parameters_")[1] for f in onlyfiles if f[:10] == "parameters"]
        return suffixes

    def import_data_for_single_session(self, session_suffix):

        parameters = pickle.load(
            open("{}/parameters_{}".format(self.folder["parameters"], session_suffix), mode='rb'))
        # print("parameters", parameters)

        indirect_exchanges = pickle.load(
            open("{}/indirect_exchanges_{}".format(self.folder["exchanges"], session_suffix), mode='rb'))

        direct_exchanges = pickle.load(
            open("{}/direct_exchanges_{}".format(self.folder["exchanges"], session_suffix), mode='rb'))

        return {"parameters": parameters, "indirect_exchanges": indirect_exchanges,
                "direct_exchanges": direct_exchanges}


def main():

    data_folder = "../single_shot_data"

    data_importer = DataImporter(data_folder=data_folder)

    suffixes = data_importer.import_suffix_list()

    print(suffixes)