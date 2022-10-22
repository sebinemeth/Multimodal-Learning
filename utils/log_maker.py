import os
from time import sleep
from numpy import ndarray as np_array
from termcolor import colored


class LogMaker(object):
    def __init__(self):
        self.log_dir_path = None
        self.log_files = list()
        self.log_id = -1
        self.current_log_id = 0

    def make_log_dir(self, log_dir_path):
        self.log_dir_path = log_dir_path
        os.makedirs(self.log_dir_path, exist_ok=True)

    def get_log_id(self):
        self.log_id += 1
        return self.log_id

    def write_log(self, file_name, log_message):
        if file_name in self.log_files:
            log_file_path = os.path.join(self.log_dir_path, file_name)
            with open(log_file_path, 'a') as log_file:
                log_file.write(log_message + '\n')
        else:
            self.log_files.append(file_name)
            log_file_path = os.path.join(self.log_dir_path, file_name)
            with open(log_file_path, 'w') as log_file:
                log_file.write(log_message + '\n')


log_maker = None


def start_log_maker():
    global log_maker
    log_maker = LogMaker()


def set_log_dir_path(log_dir_path):
    if log_maker is None:
        print(colored("Log maker aren't started yet, so base path will not be set..", color="red"))
        return
    # assert log_maker is not None, "Log maker aren't started yet."
    log_maker.make_log_dir(log_dir_path)


def write_log(file_name, log_message, title: str = None, blank_line: bool = True, separate_into_lines: bool = True,
              print_out: bool = False, color: str = None):
    """
    file_name: name of the log file, where the log message gets
    log_message: the string, list, tuple or string, that is the log message
    title: title of the message, it will be writen with capital before the message
    blank_line: put one blank line after the message
    separate_into_line: if True then list, tuple and dict will be writen more lines
    """
    if log_maker is None:
        return

    if print_out:
        print(colored(log_message, color=color))

    log_id = log_maker.get_log_id()

    while log_id != log_maker.current_log_id:
        sleep(0.1)

    if isinstance(log_message, (str, int, float, complex)) or not separate_into_lines:
        if title is not None:
            log_maker.write_log(file_name, title.upper())
        log_maker.write_log(file_name, str(log_message))
    elif isinstance(log_message, (list, tuple, np_array)):
        if title is not None:
            log_maker.write_log(file_name, title.upper())
        for message in log_message:
            log_maker.write_log(file_name, str(message))
    elif isinstance(log_message, dict):
        if title is not None:
            log_maker.write_log(file_name, title.upper())
        for key, value in log_message.items():
            message = str(key) + ': ' + str(value)
            log_maker.write_log(file_name, message)
    else:
        raise TypeError("Not allow to write {} type into the log!".format(type(log_message)))

    if blank_line:
        log_maker.write_log(file_name, '')

    log_maker.current_log_id += 1

