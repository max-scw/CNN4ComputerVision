import numpy as np
import pandas as pd
from PIL import Image
import pathlib as pl
import random
import re
from datetime import datetime
import logging
from typing import Union, Tuple

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# https://keras.io/api/applications/
from keras.applications import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    InceptionV3,
    InceptionResNetV2,
    MobileNet,
    MobileNetV2,
    MobileNetV3Small,
    MobileNetV3Large,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet50V2,
    ResNet101V2,
    ResNet152V2,
    VGG16,
    VGG19,
    Xception
)

from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay, PiecewiseConstantDecay
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Dense
from keras.models import Model, load_model
from keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives

from keras.preprocessing.image import ImageDataGenerator


class StayAliveLoggingCallback(Callback):
    _log_file_name = None
    _log_epoch_step = 1
    _external_info = ""

    def __init__(self, log_file_name: str = None, epoch_step: int = 1, info: str = ""):
        super().__init__()
        self._log_file_name = log_file_name
        self._log_epoch_step = epoch_step
        self._external_info = info

        # turn logging o
        if self._log_file_name:
            logging.basicConfig(filename=self._log_file_name, level=logging.INFO)

    def on_epoch_end(self, epoch, logs=None):
        if self._log_file_name and (epoch % self._log_epoch_step) == 0:
            logging.info(f"{datetime.now()} {self._external_info}, epoch {epoch}: "
                         f"loss {round(logs['loss'], 3)}, accuracy {round(logs['accuracy'], 3)}.")


class TrainModels:
    # local/private variables
    paths = None
    _path_to_data = None

    _random_seed = None
    _file_extension = "jpg"
    _log_file_name = None
    __split_names = {"training": "Trn", "validation": "Val", "test": "Tst"}
    __batch_size_max = 64
    __batch_size_min = 10
    _n_dense_layers_for_new_head = 1
    _stay_alive_logging_step = 50

    n_classes = None
    img_size = None
    color_mode = "rgb"
    epochs = None
    verbose = False
    training_history = None
    learning_rate = None
    use_model_checkpoints = False

    model_pretrained = False
    model_name = None

    def __init__(
            self,
            path_to_data: Union[pl.Path, str],
            epochs: int,
            random_seed: int = 42,
            path_to_save_models: Union[str, pl.Path] = None,
            verbose: bool = False,
            file_extension: str = "jpg",
            log_file_name: str = "log",
            use_model_checkpoints: bool = False
    ) -> None:
        # set local variables
        self.set_path_to_data(path_to_data, path_to_save_models)
        self.epochs = epochs
        self._random_seed = random_seed
        self.verbose = verbose
        self._file_extension = "." + file_extension.replace(".", "")
        self._log_file_name = log_file_name + ".txt"
        self.use_model_checkpoints = use_model_checkpoints

        # set classes
        self.get_n_classes()

        # turn logging on
        logging.basicConfig(filename=self._log_file_name, level=logging.INFO)

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

    def get_n_classes(self) -> int:
        if self.n_classes is None:
            dirs = []
            for ky in self.__split_names.keys():
                dirs += [el.name for el in self.paths[ky].iterdir() if el.is_dir()]
            self.n_classes = len(np.unique(dirs))
        # return number of classes
        return self.n_classes

    @property
    def get_image_size(self) -> Tuple:
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

    def get_target_size(self) -> Tuple:
        if self.model_pretrained:
            target_size = (224, 224)
        else:
            target_size = self.get_image_size

        return target_size

    def get_data_generator(self, key: str):
        args = {"directory": self.paths[key],
                "target_size": self.get_target_size(),
                "color_mode": self.color_mode,
                # "class_mode": None if key[0].upper() == 'V' else "categorical",
                "class_mode": "categorical",
                "batch_size": self.get_batch_size(key)
                }

        def add_noise(img: np.ndarray) -> np.ndarray:
            deviation = 42 * random.random()  # variability = 42
            noise = np.random.normal(0, deviation, img.shape)
            img += noise
            np.clip(img, 0.0, 255.0)
            return img

        def preproc(img: np.ndarray, color_mode: str) -> Image:
            img = add_noise(img)
            if not re.match("RGB", color_mode, re.IGNORECASE):
                img = np.expand_dims(np.mean(img, axis=2), 2)
            return img

        if re.match("train(ing)?", key, re.IGNORECASE):
            gen = ImageDataGenerator(
                rescale=1.0 / 255,
                width_shift_range=[-0.1, 0.1],
                height_shift_range=[-0.1, 0.1],
                rotation_range=10,
                vertical_flip=True,
                horizontal_flip=True,
                brightness_range=[0.2, 2.0],
                preprocessing_function=lambda x: preproc(x, self.color_mode),
                zoom_range=[0.9, 1.1],
            ).flow_from_directory(
                **args,
                shuffle=True,
                seed=self._random_seed,
            )
        else:
            gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(**args)
        return gen

    def get_data_generator_labels(self, key: str = "training", as_int: bool = True) -> list:
        gen = self.get_data_generator(key)

        gen.reset()
        y_act = []
        for i in range(int(len(gen.classes) / gen.batch_size)):
            _, y = gen.next()
            y_act += np.argmax(y, axis=1).tolist()

        if not as_int:
            y_act_names = []
            for el in y_act:
                for key, val in gen.class_indices.items():
                    if el == val:
                        break
                y_act_names.append(key)
            y_act = y_act_names
        return y_act

    def get_batch_size(self, key: str) -> int:
        n_files = self.get_n_files(key)

        batch_size = -1
        remainder = 9999
        # find most suitable batch size
        batch_size_max = np.min([self.__batch_size_max, n_files])
        for sz in range(batch_size_max, self.__batch_size_min - 1, -1):
            tmp_remainder = n_files % sz
            if tmp_remainder < remainder:
                remainder = tmp_remainder
                batch_size = sz
                if remainder == 0:
                    break
        return batch_size

    def get_n_files(self, key: str) -> int:
        n_files = 0
        files = self.paths[key].glob("*/*" + self._file_extension)
        for n_files, _ in enumerate(files):
            pass
        return n_files + 1

    def get_steps_per_epoch(self, key: str) -> int:
        return int(np.floor(self.get_n_files(key) / self.get_batch_size(key)))

    # ----- Models
    def _compile_model(self) -> bool:
        if self.learning_rate is None:
            # default constant learning rate for Adam: 0.001 = 1e-3
            if self.model_pretrained:
                self.learning_rate = 1e-4
            else:
                self.learning_rate = PiecewiseConstantDecay(boundaries=[50, 250, 1000],
                                                            values=[0.005, 0.001, 0.0005, 0.0001])
                # learning_rage = ExponentialDecay(initial_learning_rate=0.045, decay_steps=20, decay_rate=0.94)

        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss="categorical_crossentropy",
                           metrics=["accuracy", 
                                    "Precision", 
                                    "Recall", 
                                    "AUC", 
                                    TruePositives(name='tp'), 
                                    FalsePositives(name='fp'), 
                                    TrueNegatives(name='tn'), 
                                    FalseNegatives(name='fn'),])

        logging.info(
            f"{datetime.now()}: learning_rate = {self.learning_rate if isinstance(self.learning_rate, float) else self.learning_rate.get_config()}")
        return True

    def fit(self):
        # set random seed | necessary for the nested function in the data generator which produces
        random.seed(self._random_seed)
        # start training
        logging.info(f"{datetime.now()}: Start training {self.model_name} ...")
        print(f"Training {self.model_name} for {self.epochs} epochs")

        assert self.model.output.shape[1] == self.get_n_classes()

        history = self.model.fit(
            x=self.get_data_generator("training"),
            steps_per_epoch=self.get_steps_per_epoch("training"),
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=self.__callbacks(),
            workers=2,
            validation_data=self.get_data_generator("validation"),
            validation_batch_size=self.get_batch_size("validation")
        )

        self.training_history = pd.DataFrame(history.history)
        logging.info(f"{datetime.now()}: done: {self.model_name}: {self.training_history.iloc[-1].to_json()}.")
        return self.model

    def __callbacks(self) -> list:
        callbacks = [EarlyStopping(monitor="val_loss",
                                   patience=100,
                                   restore_best_weights=True,
                                   verbose=self.verbose),
                     StayAliveLoggingCallback(log_file_name=self._log_file_name,
                                              epoch_step=self._stay_alive_logging_step,
                                              info=self.__get_model_name())]

        if self.use_model_checkpoints:
            path_model_checkpoints = self.paths["models"].joinpath(self.model_name)
            path_model_checkpoints.mkdir()
            callbacks += ModelCheckpoint(filepath=path_model_checkpoints,
                                         monitor='val_loss',
                                         mode='min',
                                         save_best_only=True)
        return callbacks

    def analyze(self, model_name: str, learning_rate: Union[float, ExponentialDecay, PiecewiseConstantDecay] = None) -> Model:
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
        return file_name

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
            logging.info(f"{datetime.now()}: Model saved to {file_path}.")
            # save history to file
            file_path = self.paths["models"].joinpath(file_name + ".csv")
            self.training_history.to_csv(file_path, index=False)
        return True

    def __add_new_model_head(self) -> Model:
        base_model = self.model
        x = base_model.output
        for i in range(self._n_dense_layers_for_new_head):
            x = Dense(1024, activation="ReLU")(x)
        model_head = Dense(self.n_classes, activation="softmax")(x)
        self.model = Model(inputs=base_model.input, outputs=model_head)
        return self.model

    def _general_model_parameters(self) -> dict:
        args = {"classes": self.get_n_classes()}
        # pretrained weights
        if self.model_pretrained:
            args["weights"] = "ImageNet".lower()
            self.img_size = (224, 224)
            self.color_mode = "RGB".lower()
            if self.n_classes != 1000:
                args["include_top"] = False
                args["pooling"] = "avg"
        else:
            args["weights"] = None

        # input shape
        if re.match("RGB", self.color_mode, re.IGNORECASE):
            args["input_shape"] = self.get_target_size() + (3,)
        else:  # color_mode == grayscale
            args["input_shape"] = self.get_target_size() + (1,)

        logging.info(f"{datetime.now()}: Parameters for training: {args}.")
        return args

    def __match_model_name(self, model_name: str, input: str) -> bool:
        if re.match(model_name, input, re.IGNORECASE):
            self.model_name = model_name
            return True
        return False

    def set_model(self, model_name: str) -> Model:
        # set flag for using pretrained weights
        if re.search(r"[\s,-_\.]pretrain(ed)?", model_name, re.IGNORECASE):
            self.model_pretrained = True
        else:
            self.model_pretrained = False

        # set color mode
        if re.search(r"[\s,-_\.]gray", model_name, re.IGNORECASE):
            self.color_mode = "grayscale"
            if self.model_pretrained:
                raise ValueError('Cannot use grayscale images with pretrained weights.')
        else:
            self.color_mode = "RGB".lower()

        args = self._general_model_parameters()

        if re.match("DenseNet\d", model_name):
            # DenseNet121, DenseNet169, DenseNet201

            if self.__match_model_name("DenseNet121", model_name):
                self.model = DenseNet121(**args)
            elif self.__match_model_name("DenseNet169", model_name):
                self.model = DenseNet169(**args)
            elif self.__match_model_name("DenseNet201", model_name):
                self.model = DenseNet201(**args)
        elif re.match("Inception((V3)|(ResNetV2))", model_name):
            # InceptionV3, InceptionResNetV2

            if self.__match_model_name("InceptionV3", model_name):
                self.model = InceptionV3(**args)
            elif self.__match_model_name("InceptionResNetV2", model_name):
                self.model = InceptionResNetV2(**args)
        elif re.match("MobileNet(V[23])?((Small)|(Large))?", model_name):
            # MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large

            if self.__match_model_name("MobileNetV2", model_name):
                self.model = MobileNetV2(**args)
            elif self.__match_model_name("MobileNetV3Small", model_name):
                self.model = MobileNetV3Small(**args)
            elif self.__match_model_name("MobileNetV3Large", model_name):
                self.model = MobileNetV3Large(**args)
            elif self.__match_model_name("MobileNet", model_name):
                self.model = MobileNet(**args)
        elif re.match("ResNet((50)|(101)|(152))(V2)?", model_name):
            # ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2

            if self.__match_model_name("ResNet50", model_name):
                self.model = ResNet50(**args)
            elif self.__match_model_name("ResNet101", model_name):
                self.model = ResNet101(**args)
            elif self.__match_model_name("ResNet152", model_name):
                self.model = ResNet152(**args)
            elif self.__match_model_name("ResNet50V2", model_name):
                self.model = ResNet50V2(**args)
            elif self.__match_model_name("ResNet101V2", model_name):
                self.model = ResNet101V2(**args)
            elif self.__match_model_name("ResNet152V2", model_name):
                self.model = ResNet152V2(**args)
        elif re.match("VGG((16)|(19))", model_name):
            # VGG16, VGG19

            if self.__match_model_name("VGG16", model_name):
                self.model = VGG16(**args)
            elif self.__match_model_name("VGG19", model_name):
                self.model = VGG19(**args)
        elif self.__match_model_name("Xception", model_name):
            # Xception

            self.model = Xception(**args)
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

        if self.model_pretrained:
            self.__add_new_model_head()
            self.model_name += "_pretrained"

        # compile model
        self._compile_model()
        return self.model

    def predict(self, key: str = "test"):
        # for p2mdl in self.paths["models"].glob("*.h5"):
        #     model = load_model(p2mdl)

        gen = self.get_data_generator(key)

        y_prd_softmax = self.model.predict(x=gen,
                                           batch_size=self.get_batch_size(key),
                                           verbose=self.verbose
                                           )

        y_prd = np.argmax(y_prd_softmax, axis=1)
        y_act = gen.classes
        return pd.Series(y_prd, name="y_prd"), pd.Series(y_act, name="y_act")


if __name__ == "__main__":
    path_to_data_folder = pl.Path("Data_RGB")
    path_to_save_models = pl.Path("Models")

    # TODO: [el + '-pretrained' for el in models_to_analyze]
    # TODO: [el + '-grayscale' for el in models_to_analyze]
    models_to_analyze = ["MobileNetV2-grayscaled"]

    train = TrainModels(path_to_data_folder,
                        epochs=2,
                        verbose=True,
                        log_file_name="TestLog")
    for mdl in models_to_analyze:
        train.analyze(mdl)
    print('done.')