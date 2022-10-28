import pandas as pd
from typing import Union

from keras.callbacks import Callback
import logging
from datetime import datetime


# (45, 66, 117)     "#2D4275"   # Voith blue
# (40, 185, 218)    "#28B9DA"   # Voith cyan
# (31, 130, 192)    "#1F82C0"   # Voith light blue
# (137, 186, 23)    "#89BA17"   # Voith green
# (233, 96, 145)    "#E96091"   # Voith pink
# (129, 142, 172)   "#818EAC"   # Voith silver
voith_colors = pd.DataFrame([
    {"name": "voith_blue", "RGB": (45, 66, 117), "hex": "#2D4275"},
    {"name": "voith_cyan", "RGB": (40, 185, 218), "hex": "#28B9DA"},
    # light-blue
    {"name": "voith_green", "RGB": (137, 186, 23), "hex": "#89BA17"},
    {"name": "voith_pink", "RGB": (233, 96, 145), "hex": "#E96091"},
    {"name": "voith_silver", "RGB": (129, 142, 172), "hex": "#818EAC"},
    {"name": "voith_lightblue", "RGB": (31, 130, 192), "hex": "#1F82C0"},
]).set_index("name")

# def voith_colors(index: Union[int, str], format: str = "RGB"):
#     pass


class Log:
    _filename = None

    def __init__(self, file_name: str = None, print_message: bool = False):
        if file_name:
            self._filename = file_name.strip("*.")
            # turn logging on
            logging.basicConfig(filename=self._filename + ".log", level=logging.INFO)
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
        self._log_file_name = log_file_name
        self._log_epoch_step = epoch_step
        self._external_info = info
        self._log = Log(file_name=log_file_name, print_message=False)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self._log_epoch_step) == 0:
            msg = ""
            if self._external_info:
                msg += f"{self._external_info} , "
            msg += f"epoch {epoch}: " +\
                  ", ".join([f"{ky} {round(logs[ky], 4)}" for ky in logs.keys()])
            self._log.log(msg)
