from pylab import np, plt
from os import mkdir
from os.path import exists
from tqdm import tqdm
from module.data_importer import DataImporter


class Analysis(object):

    @classmethod
    def simple_analysis(cls, indirect_exchange, figure_folder, msg="", suffix=""):

        cls.plot(indirect_exchange, figure_folder, msg=msg, suffix=suffix)

    @classmethod
    def plot(cls, data, figure_folder, msg="", suffix=""):

        x = np.arange(len(data[:]))

        plt.plot(x, data[:, 0], c="red", linewidth=2)
        plt.plot(x, data[:, 1], c="blue", linewidth=2)
        plt.plot(x, data[:, 2], c="green", linewidth=2)
        plt.ylim([-0.01, 1.01])
        plt.text(0, -0.1, "{}".format(msg))

        if not exists(figure_folder):
            mkdir(figure_folder)
        fig_name = "{}/figure_{}.pdf".format(figure_folder, suffix.split(".p")[0])
        plt.savefig(fig_name)
        plt.close()


def main():

    data_folder = "../single_shot_data"
    figure_folder = "../single_shot_figures"

    data_importer = DataImporter(data_folder=data_folder)

    suffixes = data_importer.import_suffix_list()

    for suffix in tqdm(suffixes):

        data = data_importer.import_data_for_single_session(suffix)
        msg = "Workforce: {}, alpha: {}, tau: {}, epsilon: {}, q_info: {}"\
            .format(data["parameters"]["workforce"],
                    data["parameters"]["alpha"],
                    data["parameters"]["tau"],
                    data["parameters"]["epsilon"],
                    data["parameters"]["q_information"]
                    )

        Analysis.simple_analysis(data["indirect_exchanges"], figure_folder=figure_folder, msg=msg, suffix=suffix)


if __name__ == "__main__":

    # Analysis.simple_analysis(results_folder="../data", session_suffix="example_idx0")
    main()

