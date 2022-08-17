from datetime import datetime
import logging
import numpy as np
from PIL import Image
import pathlib as pl
import random
import re
from typing import Union, Tuple

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# MobileNet(s)
from keras.applications import (
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
    Xception,
)

# https://keras.io/api/applications/

from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator


class TrainModels:
    # local/private variables
    paths = None
    _path_to_data = None

    _random_seed = None
    _file_extension = "jpg"
    __split_names = {"training": "Trn", "validation": "Val", "test": "Tst"}
    __log_file_name = "log.out"
    __batch_size_max = 30
    __batch_size_min = 10
    _n_dense_layers_for_new_head = 1

    n_classes = None
    img_size = None
    color_mode = "rgb"
    epochs = None
    verbose = False
    training_history = None

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
    ) -> None:
        # set local variables
        self.set_path_to_data(path_to_data, path_to_save_models)
        self.epochs = epochs
        self._random_seed = random_seed
        self.verbose = verbose
        self._file_extension = "." + file_extension.replace(".", "")

        # set classes
        self.get_n_classes()
        # set data generator
        self.get_data_generator_instance()

        # turn logging on
        logging.basicConfig(filename=self.__log_file_name, level=logging.INFO)

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

    def get_image_size(self) -> Tuple:
        if self.img_size is None:
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

    def get_data_generator(self, key: str):
        args = {"directory": self.paths[key],
                "target_size": self.img_size,
                "color_mode": self.color_mode,
                "class_mode": "categorical",
                "batch_size": self.get_batch_size(key)
                }

        def __add_noise(img: np.ndarray) -> np.ndarray:
            deviation = 50 * random.random()  # variability = 50
            noise = np.random.normal(0, deviation, img.shape)
            img += noise
            np.clip(img, 0.0, 255.0)
            return img

        if re.match("train(ing)?", key, re.IGNORECASE):
            gen = ImageDataGenerator(
                rescale=1.0 / 255,
                width_shift_range=[-0.2, 0.2],
                height_shift_range=[-0.2, 0.2],
                rotation_range=10,
                vertical_flip=True,
                horizontal_flip=True,
                brightness_range=[0.2, 2.0],
                preprocessing_function=__add_noise,
                zoom_range=[0.9, 1.1],
            ).flow_from_directory(
                **args,
                shuffle=True,
                seed=self._random_seed,
            )
        else:
            gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(**args)
        return gen

    def get_batch_size(self, key: str) -> int:
        n_files = self.get_n_files(key)

        batch_size = -1
        remainder = 9999
        # find most suitable batch size
        for sz in range(self.__batch_size_max, self.__batch_size_min - 1, -1):
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
        return n_files

    def get_steps_per_epoch(self, key: str) -> int:
        return int(np.floor(self.get_n_files(key) / self.get_batch_size(key)))

    # ----- Models
    def _compile_model(self) -> bool:
        self.model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
        return True

    def fit(self):
        # set random seed | necessary for the nested function in the data generator which produces
        random.seed(self._random_seed)
        # start training
        logging.info(f"{datetime.now()}: Start training {self.model_name} ...")
        self.training_history = self.model.fit(
            x=self.get_data_generator("training"),
            steps_per_epoch=self.get_steps_per_epoch("training"),
            epochs=self.epochs,
            verbose=self.verbose,
        )
        if self.epochs > 1:
            final_metrics = {ky: self.training_history.history[ky][-1] for ky in self.training_history.history}
        else:
            final_metrics = self.training_history.history
        logging.info(f"{datetime.now()}: done. \n {self.model_name}: {final_metrics}")
        return self.model

    def analyze(self, model_name: str) -> Model:
        # set model
        self.set_model(model_name)
        # train model
        self.fit()
        # save model
        if self.paths["models"]:
            file_path = self.paths["models"].joinpath(f"{model_name}.h5")
            self.model.save(file_path)
            logging.info(f"{datetime.now()}: Model saved to {file_path}.")
        return self.model

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
            args["weights"] = "ImageNet"
            self.img_size = (224, 224)
            self.color_mode = "RGB"
            if self.n_classes != 1000:
                args["include_top"] = False
                args["pooling"] = "avg"
        else:
            args["weights"] = None

        # input shape
        if re.match("RGB", self.color_mode, re.IGNORECASE):
            args["input_shape"] = self.img_size + (3,)
        else:  # color_mode == grayscale
            args["input_shape"] = self.img_size + (1,)

        logging.info(f"{datetime.now()}: Parameters for training: {args}.")
        return args

    def _match_model_name(self, model_name: str, input: str) -> bool:
        if re.match(model_name, input, re.IGNORECASE):
            self.model_name = model_name
            return True
        return False

    def set_model(self, model_name: str) -> Model:
        if re.search("pretrain(ed)?", model_name, re.IGNORECASE):
            self.model_pretrained = True
        else:
            self.model_pretrained = False

        args = self._general_model_parameters()

        if re.match("Inception((V3)|(ResNetV2))", model_name):
            # InceptionV3, InceptionResNetV2

            if self._match_model_name("InceptionV3", model_name):
                self.model = InceptionV3(**args)
        elif re.match("MobileNet(V[23])?((Small)|(Large))?", model_name):
            # MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large

            if self._match_model_name("MobileNet", model_name):
                self.model = MobileNet(**args)
            elif self._match_model_name("MobileNetV2", model_name):
                self.model = MobileNetV2(**args)
            elif self._match_model_name("MobileNetV3Small", model_name):
                self.model = MobileNetV3Small(**args)
            elif self._match_model_name("MobileNetV3Large", model_name):
                self.model = MobileNetV3Large(**args)
        elif re.match("ResNet((50)|(101)|(152))(V2)?", self.model_name):
            # ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2

            if self._match_model_name("ResNet50", model_name):
                self.model = ResNet50(**args)
            elif self._match_model_name("ResNet101", model_name):
                self.model = ResNet101(**args)
            elif self._match_model_name("ResNet152", model_name):
                self.model = ResNet152(**args)
            elif self._match_model_name("ResNet50V2", model_name):
                self.model = ResNet50V2(**args)
            elif self._match_model_name("ResNet101V2", model_name):
                self.model = ResNet101V2(**args)
            elif self._match_model_name("ResNet152V2", model_name):
                self.model = ResNet152V2(**args)
        elif re.match("VGG((16)|(19))", model_name):
            # VGG16, VGG19

            if self._match_model_name("VGG16", model_name):
                self.model = VGG16(**args)
            elif self._match_model_name("VGG19", model_name):
                self.model = VGG19(**args)
        elif self._match_model_name("Xception", model_name):
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


if __name__ == "__main__":
    path_to_data_folder = pl.Path("Data")
    path_to_save_models = pl.Path("Models")

    models_to_analyze = ["MobileNet", "MobileNetV2", "InceptionV3"]

    train = TrainModels(path_to_data_folder, epochs=2, verbose=True)
    for mdl in models_to_analyze:
        train.analyze(mdl)