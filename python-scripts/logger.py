

import sys
import os

# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting

class Logger(object):
    def __init__(self, logdir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(logdir, "run.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

