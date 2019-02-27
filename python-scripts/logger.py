

import sys
import os

# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting




def set_redirects(logdir):
    sys.stdout = Logger(logdir, sys.stdout)
    sys.stderr = Logger(logdir, sys.stderr)

class Logger(object):
    def __init__(self, logdir, std):
        self.terminal = std
        self.log = open(os.path.join(logdir, "run.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

