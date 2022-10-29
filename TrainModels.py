import numpy as np
import pandas as pd
from PIL import Image
import pathlib as pl
import random
import re
from datetime import datetime
from typing import Union, Tuple

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay, PiecewiseConstantDecay
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model, load_model
from keras import backend as K

from utils import StayAliveLoggingCallback, Log


class TrainModels:
    # local/private variables
    paths = None
    _path_to_data = None

    _random_seed = None
    _file_extension = "jpg"
    _log_file_name = None
    __split_names = {"training": "Trn", "validation": "Val", "test": "Tst"}
    _n_dense_layers_for_new_head = 1
    _stay_alive_logging_step = 5
    __early_stopping_after_n_epochs = 42

    n_classes = None
    img_size = None
    color_mode = "rgb"
    epochs = None
    verbose = False
    training_history = None
    learning_rate = None
    use_model_checkpoints = False

    model_name = None
    model_pretrained = False
    model_finetune = False
    model = None

    def __init__(
            self,
            path_to_data: Union[pl.Path, str],
            epochs: int,
            random_seed: int = 42,
            path_to_save_models: Union[str, pl.Path] = None,
            verbose: bool = False,
            file_extension: str = "jpg",
            log_file_name: str = None,
            use_model_checkpoints: bool = False,
            n_classes: int = None
    ) -> None:
        # set local variables
        self.set_path_to_data(path_to_data, path_to_save_models)
        self.epochs = epochs
        self._random_seed = random_seed
        self.verbose = verbose
        self._file_extension = "." + file_extension.replace(".", "")
        self.use_model_checkpoints = use_model_checkpoints

        # set classes
        self.n_classes = n_classes
        self.n_examples = {ky: self.get_n_files(ky) for ky in self.__split_names.keys()}

        # turn logging on
        self._log = Log(file_name=log_file_name, print_message=self.verbose)

    def set_path_to_data(
            self, path_to_data: Union[pl.Path, str], path_to_save_models: Union[pl.Path, str, None]
    ) -> bool:
        # TODO: check for correct folder structure!
        self._path_to_data = pl.Path(path_to_data)

        self.paths = dict()
        for el in self.__split_names.items():
            self.paths[el[0]] = self._path_to_data.joinpath(el[1])
        if path_to_save_models:
            self.paths["models"] = path_to_save_models
            # create directory if necessary
            self.paths["models"].mkdir(parents=False, exist_ok=True)

        return True

    @property
    def image_size(self) -> Tuple:
        p2file = None
        # find all images
        for p2file in self._path_to_data.glob("**/*" + self._file_extension):
            if p2file:
                break
        # read image
        if p2file:
            img = Image.open(p2file)
            self.img_size = img.size
        else:
            raise ValueError("No images found to infer the images size from.")
        # return image size
        return self.img_size

    # def get_data_generator(self, key: str, shuffle: bool = True):
    #     gen = None
    #     return gen

    def get_batch_size(self, key: str):
        return self._determine_batch_size(self.get_n_files(key))

    @staticmethod
    def _determine_batch_size(n_examples: int, batch_size_max: int = 64) -> int:

        batch_size_max = np.min([batch_size_max, np.max([n_examples // 3, 1])])

        # find most suitable batch size
        batch_size = -1
        remainder = np.inf
        for sz in range(batch_size_max, 0, -1):
            tmp_remainder = n_examples % sz
            if tmp_remainder < remainder:
                remainder = tmp_remainder
                batch_size = sz
                if remainder == 0:
                    break
        return batch_size

    def get_n_files(self, key: str) -> int:
        """
        count number of files
        :param key:
        :return:
        """
        n_files = 0
        files = self.paths[key].glob("**/*" + self._file_extension)
        for n_files, _ in enumerate(files):
            pass
        return n_files + 1

    def get_steps_per_epoch(self, key: str) -> int:
        return self.get_n_files(key) // self.get_batch_size(key)

    # def _compile_model(self, finetune: bool = False) -> bool:
    #     # TODO
    #     if self.learning_rate is None:
    #         # default constant learning rate for Adam: 0.001 = 1e-3
    #         if finetune:
    #             self.learning_rate = ExponentialDecay(initial_learning_rate=1e-4, decay_steps=20, decay_rate=0.9)
    #         elif self.model_pretrained:
    #             self.learning_rate = 1e-3
    #         else:
    #             self.learning_rate = PiecewiseConstantDecay(boundaries=[50, 250, 1000],
    #                                                         values=[0.005, 0.001, 0.0005, 0.0001])
    #             # learning_rage = ExponentialDecay(initial_learning_rate=0.045, decay_steps=20, decay_rate=0.94)
    #
    #     self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
    #                        loss="categorical_crossentropy",
    #                        metrics=["accuracy",
    #                                 "Precision",
    #                                 "Recall",
    #                                 "AUC",
    #                                 TruePositives(name='tp'),
    #                                 FalsePositives(name='fp'),
    #                                 TrueNegatives(name='tn'),
    #                                 FalseNegatives(name='fn')
    #                                 ])
    #
    #     logging.info(
    #         f"{datetime.now()}: learning_rate = {self.learning_rate if isinstance(self.learning_rate, float) else self.learning_rate.get_config()}"
    #     )
    #     return True

    def fit(self):
        # K.clear_session()
        self._compile_model()
        model = self._fit()
        # if finetuning is activated, a second training step is required
        if self.model_finetune:
            self.model.trainable = True
            self._compile_model(finetune=True)
            model = self._fit()
        return model

    def _fit(self):
        # set random seed | necessary for the nested function in the data generator which produces
        random.seed(self._random_seed)
        # start training
        self._log.log(f"Start training {self.model_name} for {self.epochs} epochs ...")

        # assert self.model.output.shape[1] == self.n_classes

        history = self.model.fit(
            x=self.get_data_generator("training"),
            steps_per_epoch=self.get_steps_per_epoch("training"),
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=self.__callbacks(),
            validation_data=self.get_data_generator("validation"),
            validation_steps=self.get_steps_per_epoch("validation")
        )

        self.training_history = pd.DataFrame(history.history)
        self._log.log(f"done. {self.model_name}: {self.training_history.iloc[-1].to_json()}.")
        return self.model

    def __callbacks(self) -> list:
        callbacks = [EarlyStopping(monitor="val_loss",
                                   patience=self.__early_stopping_after_n_epochs,
                                   restore_best_weights=True,
                                   verbose=self.verbose
                                   ),
                     ReduceLROnPlateau(monitor="val_loss",
                                       factor=0.75,
                                       patience=25,
                                       min_lr=1e-8,
                                       min_delta=1e-6,
                                       cooldown=15,
                                       verbose=self.verbose
                                       ),
                     StayAliveLoggingCallback(log_file_name=self._log_file_name,
                                              epoch_step=self._stay_alive_logging_step,
                                              info=self.__get_model_name()
                                              )
                     ]

        if self.use_model_checkpoints:
            path_model_checkpoints = self.paths["models"].joinpath(self.model_name)
            path_model_checkpoints.mkdir(parents=False, exist_ok=True)
            callbacks += [ModelCheckpoint(filepath=path_model_checkpoints,
                                          monitor='val_loss',
                                          mode='min',
                                          save_best_only=True,
                                          verbose=self.verbose
                                          )]
        return callbacks

    def _save_model(self) -> bool:
        if "models" in self.paths and self.paths["models"]:
            # create file name
            date_str = datetime.now().strftime(format="%y%m%d%H%M")
            file_name = f"{date_str}_{self.__get_model_name()}"
            # build file path
            file_path = self.paths["models"].joinpath(file_name + ".h5")
            # save model to file
            self.model.save(file_path)
            # log
            msg = f"Model {self.model_name} saved to {file_path}."
            self._log.log(msg, print_message=True)
            # save history to file
            file_path = self.paths["models"].joinpath(file_name + ".csv")
            self.training_history.to_csv(file_path, index=False)
        return True

    def analyze(self, model_name: str,
                learning_rate: Union[float, ExponentialDecay, PiecewiseConstantDecay] = None) -> Model:
        # set learning rate
        self.learning_rate = learning_rate
        # set model
        self.set_model(model_name)
        # train model
        self.fit()
        # save model
        self._save_model()
        return self.model

    def __get_model_name(self) -> str:
        file_name = f"{self.model_name}_{self.color_mode}"
        if self.model_pretrained:
            file_name += "-pretrained"  # TODO: make optional to train on what weights
        if self.model_finetune:
            file_name += "-finetuned"
        return file_name

    def _set_flags_from_model_name(self, model_name: str) -> bool:
        pattern_add = r"[,_\s\-\.\+\\\|]"
        # set flag for using pretrained weights
        if re.search(pattern_add + "pretrain(ed)?", model_name, re.IGNORECASE):
            self.model_pretrained = True
            # set flag for finetuninng (pretrained) weights
            if re.search(pattern_add + "finetun(e|(ing))", model_name, re.IGNORECASE):
                self.model_finetune = True
            else:
                self.model_finetune = False
        else:
            self.model_pretrained = False

        # set color mode
        if re.search(pattern_add + "gray", model_name, re.IGNORECASE):
            self.color_mode = "grayscale"
            if self.model_pretrained:
                raise ValueError('Cannot use grayscale images with pretrained weights.')
        else:
            self.color_mode = "RGB".lower()
        return True

    def _match_model_name(self, model_name: str, input: str) -> bool:
        if re.match(model_name, input, re.IGNORECASE):
            self.model_name = model_name
            return True
        return False

    # def set_model(self, model_name: str) -> Model:
    #     self.model = None
    #     return self.model
