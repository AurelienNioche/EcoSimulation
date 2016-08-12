# -*- coding: utf-8 -*-
from module.save import Cursor, BackUp
from multiprocessing import Pool
from time import time


class SimulationManager(object):

    def __init__(self, n_workers, back_up_fq=10):

        self.cursor = Cursor()
        self.pool = Pool(processes=n_workers)
        self.back_up = BackUp()

        self.back_up_fq = back_up_fq

    @staticmethod
    def convert_seconds_to_h_m_s(seconds):

        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def launch_test(self, function, arg_list):

        """
        Require a list of arguments
        :return: None
        """

        beginning_time = time()

        print("********************")

        self.cursor.retrieve_position()

        to_do = len(arg_list)

        print("Begin testing.")

        while self.cursor.position + self.back_up_fq < to_do:

            time_spent = self.convert_seconds_to_h_m_s(time() - beginning_time)

            print("Cursor position: {}/{} (time spent: {}).".format(self.cursor.position, to_do, time_spent))
            print("********************")

            results = self.pool.map(function,
                                    arg_list[self.cursor.position:self.cursor.position + self.back_up_fq])

            self.back_up.save(results)

            self.cursor.position += self.back_up_fq

            self.cursor.save_position()

        if self.cursor.position + self.back_up_fq == (to_do - 1):

            pass

        else:
            time_spent = self.convert_seconds_to_h_m_s(time() - beginning_time)

            print("Cursor position: {}/{} (time spent: {}).".format(self.cursor.position, to_do, time_spent))
            print("********************")

            results = self.pool.map(function,
                                    arg_list[self.cursor.position:])
            self.back_up.save(results)

        time_spent = self.convert_seconds_to_h_m_s(time() - beginning_time)

        print("Cursor position: {}/{} (time spent: {}).".format(to_do, to_do, time_spent))
        print("********************")

        self.cursor.reset()

        print("End of testing program.")