import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import os
import csv

class DataLog:

    def __init__(self):
        self.log = {}
        self.max_len = 0

    def get_last_iteration_no(self):
        return self.log['iteration'][-1]

    def log_kv(self, key, value):
        # logs the (key, value) pair

        # TODO: This implementation is wrong and error-prone:
        # it would be NOT aligned if some keys are missing during one iteration.
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(value)
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1

    def align_rows(self):
        for key in self.log.keys():
            while self.max_len > len(self.log[key]):
                self.log[key].append(self.log[key][-1])

    def save_log(self, save_path):
        # TODO: Validate all lengths are the same.
        pickle.dump(self.log, open(save_path + '/log.pickle', 'wb'))
        with open(save_path + '/log.csv', 'w') as csv_file:
            fieldnames = list(self.log.keys())
            if 'iteration' not in fieldnames:
                fieldnames = ['iteration'] + fieldnames

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(self.max_len):
                row_dict = {}
                for key in self.log.keys():
                    if i < len(self.log[key]):
                        row_dict[key] = self.log[key][i]
                if 'iteration' not in row_dict:
                    row_dict['iteration'] = i
                writer.writerow(row_dict)

    def get_current_log(self):
        row_dict = {}
        for key in self.log.keys():
            # TODO: this is very error-prone (alignment is not guaranteed)
            row_dict[key] = self.log[key][-1]
        return row_dict

    def shrink_to(self, num_entries):
        for key in self.log.keys():
            self.log[key] = self.log[key][:num_entries]

        assert min([len(series) for series in self.log.values()]) == \
            max([len(series) for series in self.log.values()])

    def read_log(self, log_path):
        assert log_path.endswith('log.csv')

        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row, row_dict in enumerate(listr):
                for key in keys:
                    try:
                        data[key].append(eval(row_dict[key]))
                    except:
                        print("ERROR on reading key {}: {}".format(key, row_dict[key]))

        self.log = data
        self.max_len = max(len(v) for k, v in self.log.items())
        print("Log read from {}: had {} entries".format(log_path, self.max_len))
