from os import path, listdir, rename
from tqdm import tqdm
from collections import OrderedDict
from datetime import date
from multiprocessing import Process, SimpleQueue, Event, Value
from sqlite3 import connect, OperationalError
from module.folders import Folders


class Database(object):

    def __init__(self, folder="../data", name="db"):

        self.db_path = "{}/{}.db".format(folder, name)

        self.connexion = connect(self.db_path)
        self.cursor = self.connexion.cursor()

        self.is_close = 0

    def create_table(self, table_name, columns):

        assert type(columns) == dict or type(columns) == OrderedDict, \
            "Columns type should be dict such dict or OrderedDict with as keys names and as values types in string"

        query = "CREATE TABLE `{}` (" \
                "ID INTEGER PRIMARY KEY AUTOINCREMENT, ".format(table_name)

        for key, value in columns.items():

            query += "`{}` {}, ".format(key, value)

        query = query[:-2]
        query += ")"
        self.write(query)

    def remove_table(self, table_name):

        q = "DROP TABLE `{}`".format(table_name)
        self.cursor.execute(q)

    def has_table(self, table_name):

        already_existing = self.get_tables_names()

        return table_name in already_existing

    def get_tables_names(self):

        if path.exists(self.db_path):

            # noinspection SqlResolve
            already_existing = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

            if already_existing:

                already_existing = [i[0] for i in already_existing]
                return already_existing

            else:
                return []

        else:

            return []

    def get_columns(self, table_name='data'):

        tuple_column = [(i[1], i[2]) for i in self.read("PRAGMA table_info(`{}`)".format(table_name)) if i[1] != "ID"]
        dic_column = OrderedDict()

        for i, j in tuple_column:

            dic_column[i] = j

        return dic_column

    def read(self, query):

        try:
            self.cursor.execute(query)
        except OperationalError as e:
            print("Error with query:", query)
            raise e

        content = self.cursor.fetchall()

        return content

    def write(self, query):

        try:
            self.cursor.execute(query)
        except OperationalError as e:
            print("Error with query:", query)
            raise e

    def read_n_rows(self, columns, table_name='data'):

        read_query = "SELECT "

        for i in columns.keys():
            read_query += "`{}`, ".format(i)

        read_query = read_query[:-2]
        read_query += " from `{}`".format(table_name)

        return self.read(read_query)

    def create_table_and_write_n_rows(self, columns, array_like, table_name='data'):

        assert type(columns) == dict or type(columns) == OrderedDict, \
            "Columns type should be dict such dict or OrderedDict with as keys names and as values types in string"

        create_table_query = \
            "CREATE TABLE `{}` (" \
            "ID INTEGER PRIMARY KEY, ".format(table_name)

        for key, value in columns.items():
            create_table_query += "`{}` {}, ".format(key, value)

        create_table_query = create_table_query[:-2]
        create_table_query += ")"

        fill_query = "INSERT INTO '{}' (".format(table_name)

        for i in columns.keys():

            fill_query += "`{}`, ".format(i)

        fill_query = fill_query[:-2]
        fill_query += ") VALUES ("

        for i in range(len(columns)):
            fill_query += "?, "

        fill_query = fill_query[:-2]
        fill_query += ")"

        self.cursor.execute(create_table_query)

        self.cursor.executemany(fill_query, array_like)

    def close(self):

        self.connexion.commit()
        self.connexion.close()

        self.is_close = 1

    def __del__(self):

        if not self.is_close:

            self.close()


class Writer(Process):

    def __init__(self, db_folder, db_name, queue, shutdown):

        Process.__init__(self)

        self.db = Database(folder=db_folder, name=db_name)
        self.queue = queue
        self.shutdown = shutdown
        self.counter = Value('i', 0)

    def get_tables_names(self):

        return self.db.get_tables_names()

    def run(self):

        already_existing_tables = self.db.get_tables_names()

        while not self.shutdown.is_set():

            try:

                param = self.queue.get()

                if param is not None:
                    if param[0] == "write":

                        table_name, columns, content = param[1]

                        if table_name in already_existing_tables:
                            self.db.remove_table(table_name=table_name)
                            self.db.connexion.commit()

                        self.db.create_table_and_write_n_rows(table_name=table_name, columns=columns,
                                                              array_like=content)

                        already_existing_tables.append(table_name)

                    elif param[0] == "remove_db":

                        old_db_name = param[1]
                        rename(old_db_name, "{}/{}".format(Folders.trash, old_db_name.split("/")[-1]))
                        self.counter.value += 1
                    else:
                        raise Exception("Bad argument for writer queue.")
                else:
                    break

            except KeyboardInterrupt:

                print()
                print("Writer grab keyboard interrupt.")
                if not self.db.is_close:
                    self.db.close()

        if not self.db.is_close:
            self.db.close()
        self.shutdown.set()
        print("Writer: DEAD.")


class Reader(Process):

    def __init__(self, db_folder, db_to_merge, queue, shutdown):

        Process.__init__(self)
        self.db_folder = db_folder
        self.db_to_merge = db_to_merge
        self.shutdown = shutdown
        self.writer_queue = queue

    def run(self):

        for db_name in self.db_to_merge:

            if not self.shutdown.is_set():

                db = Database(folder=self.db_folder, name=db_name)

                try:

                    db_tables_names = db.get_tables_names()

                    for table_name in db_tables_names:

                        if not self.shutdown.is_set():
                            columns = db.get_columns(table_name)
                            content = db.read_n_rows(table_name=table_name, columns=columns)
                            self.writer_queue.put(["write", [table_name, columns, content]])
                        else:
                            break

                    db.close()

                    old_db_name = "{}/{}.db".format(self.db_folder, db_name)
                    self.writer_queue.put(["remove_db", old_db_name])

                except KeyboardInterrupt:

                    print()
                    print("Reader grab keyboard interrupt.")
                    break

            else:
                break

        if not self.shutdown.is_set():
            self.writer_queue.put(None)

        print("Reader: DEAD.")


class DbManager(object):

    @staticmethod
    def merge_db(db_folder, new_db_name, db_to_merge):

        assert path.exists(db_folder), '`{}` is a wrong path to db folder, please correct it.'.format(db_folder)

        shutdown = Event()
        writer_queue = SimpleQueue()

        writer = Writer(db_folder=db_folder, db_name=new_db_name, queue=writer_queue, shutdown=shutdown)
        reader = Reader(db_folder=db_folder, db_to_merge=db_to_merge,
                        queue=writer_queue, shutdown=shutdown)

        reader.start()
        writer.start()

        pbar = tqdm(total=len(db_to_merge))

        c = 0
        while not shutdown.is_set():
            try:
                new_c = writer.counter.value
                progress = new_c - c
                if progress > 0:
                    pbar.update(progress)
                    c = new_c
                Event().wait(2)

            except KeyboardInterrupt:
                print()
                print("Main thread grab the keyboard interrupt")
                break

        shutdown.set()
        pbar.close()

        if writer.is_alive():

            print("Waiting writer...")
            writer.join()

        print("WRITER EXECUTED.")

        if reader.is_alive():
            writer_queue.get()
            print("Waiting reader...")
            reader.join()

        print("READER EXECUTED.")

        print()
        print("Done.")

    @classmethod
    def merge_all_db_from_same_folder(cls, db_folder, new_db_name):

        # Be sure that the path of the folder containing the databases is correct.
        assert path.exists(db_folder), 'Wrong path to db folder, please correct it.'

        # Get the list of all the databases
        list_db_name = [i[:-3] for i in listdir(db_folder) if i[-3:] == ".db" and i[:-3] != new_db_name]
        assert len(list_db_name), 'Could not find any db...'

        cls.merge_db(db_folder=db_folder, new_db_name=new_db_name, db_to_merge=list_db_name)

    @classmethod
    def run(cls):

        db_folder = Folders.data
        new_db_name = "data_{}".format(str(date.today()).replace("-", "_"))

        cls.merge_all_db_from_same_folder(db_folder=db_folder, new_db_name=new_db_name)


if __name__ == "__main__":

    DbManager.run()
