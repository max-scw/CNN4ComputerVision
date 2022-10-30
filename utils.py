from typing import Union

from keras.callbacks import Callback
import logging
from datetime import datetime


class Log:
    _filename = None

    def __init__(self, file_name: str = None, print_message: bool = False, level=logging.INFO):
        if file_name:
            self._filename = file_name.strip("*.") + ".log"
            # turn logging on
            logging.basicConfig(filename=self._filename, level=level)
        self._print_message = print_message

    def log(self, message: str, print_message: bool = False):
        # build message
        msg = f"{datetime.now()}: {message}"
        if self._filename:
            logging.info(msg)
        if self._print_message or print_message:
            print(msg)


class StayAliveLoggingCallback(Callback):
    _log_epoch_step = 1
    _external_info = None

    def __init__(self, log_file_name: str = None, epoch_step: int = 1, info: str = None):
        super().__init__()
        self._log_epoch_step = epoch_step
        self._external_info = info
        self._log = Log(file_name=log_file_name, print_message=False)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self._log_epoch_step) == 0:
            msg = ""
            if self._external_info:
                msg += f"{self._external_info}, "
            msg += f"epoch {epoch}: " +\
                  ", ".join([f"{ky} {round(logs[ky], 4)}" for ky in logs.keys()])
            self._log.log(msg)
