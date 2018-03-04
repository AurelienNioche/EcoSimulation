from os import path


class Folders(object):

    root_folder = "/" + path.join(*path.realpath(__file__).split("/")[:-3])
    program_folder = "/" + path.join(*path.realpath(__file__).split("/")[:-2])
    data = "{}/data".format(root_folder)
    scripts = "{}/avakas_scripts".format(root_folder)
    input_parameters = "{}/avakas_input_parameters".format(root_folder)
    logs = "{}/avakas_logs".format(root_folder)
    job_names = "{}/avakas_job_names".format(root_folder)
    trash = "{}/avakas_trash".format(root_folder)
    macro = "{}/avakas".format(program_folder)

    @classmethod
    def list(cls):

        folder_list = [j for i, j in cls.__dict__.items() if i[0] is not "_" and i != "list"]
        return folder_list
