from TrainModels import TrainModels

from typing import Union, Tuple
import pathlib as pl
import numpy as np
import pandas as pd
from PIL import Image
import random
import re
from datetime import datetime
import logging

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
from keras.layers import Dense
from keras.models import Model, load_model

from keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from keras.preprocessing.image import ImageDataGenerator


class TrainModelsImageClassification(TrainModels):
    def __init__(self,
                 path_to_data: Union[pl.Path, str],
                 epochs: int,
                 random_seed: int = 42,
                 path_to_save_models: Union[str, pl.Path] = None,
                 verbose: bool = False,
                 file_extension: str = "jpg",
                 log_file_name: str = "log",
                 use_model_checkpoints: bool = False,
                 target_size: Tuple[int, int] = (224, 224)
                 ) -> None:
        super().__init__(path_to_data=path_to_data,
                         epochs=epochs,
                         random_seed=random_seed,
                         path_to_save_models=path_to_save_models,
                         verbose=verbose,
                         file_extension=file_extension,
                         log_file_name=log_file_name,
                         use_model_checkpoints=use_model_checkpoints
                         )
        self.n_classes = self.__set_n_classes()
        self.target_size = target_size

    def __set_n_classes(self) -> int:
        if self.n_classes is None:
            dirs = []
            for ky in self.__split_names.keys():
                dirs += [el.name for el in self.paths[ky].iterdir() if el.is_dir()]
            self.n_classes = len(np.unique(dirs))
        # return number of classes
        return self.n_classes

    def get_data_generator(self, key: str, shuffle: bool = True):
        args = {"directory": self.paths[key],
                "target_size": self.target_size,
                "color_mode": self.color_mode,
                # "class_mode": None if key[0].upper() == 'V' else "categorical",
                "class_mode": "categorical",
                "batch_size": self.determine_batch_size(key)
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
                rescale=1.0 / 255,  # TODO: sometimes z-scaled images used? => MobileNet
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
                shuffle=shuffle,
                seed=self._random_seed,
            )
        else:
            gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(**args, shuffle=shuffle)
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

    # ----- Models
    def _compile_model(self, finetune: bool = False) -> bool:
        if self.learning_rate is None:
            # default constant learning rate for Adam: 0.001 = 1e-3
            if finetune:
                self.learning_rate = ExponentialDecay(initial_learning_rate=1e-4, decay_steps=20, decay_rate=0.9)
            elif self.model_pretrained:
                self.learning_rate = 1e-3
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
                                    FalseNegatives(name='fn')
                                    ])

        logging.info(
            f"{datetime.now()}: learning_rate = {self.learning_rate if isinstance(self.learning_rate, float) else self.learning_rate.get_config()}"
        )
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
        if self.model_pretrained:  # FIXME: model weights
            args["weights"] = "ImageNet".lower()
            self.target_size = (224, 224)
            self.color_mode = "RGB".lower()
            if self.n_classes != 1000:
                args["include_top"] = False
                args["pooling"] = "avg"
        else:
            args["weights"] = None

        # input shape
        if re.match("RGB", self.color_mode, re.IGNORECASE):
            args["input_shape"] = self.target_size + (3,)
        else:  # color_mode == grayscale
            args["input_shape"] = self.target_size + (1,)

        logging.info(f"{datetime.now()}: Parameters for training: {args}.")
        return args

    def set_model(self, model_name: str) -> Model:

        self._set_flags_from_model_name(model_name)
        args = self._general_model_parameters()

        if re.match(r"DenseNet\d", model_name):
            # DenseNet121, DenseNet169, DenseNet201

            if self._match_model_name("DenseNet121", model_name):
                self.model = DenseNet121(**args)
            elif self._match_model_name("DenseNet169", model_name):
                self.model = DenseNet169(**args)
            elif self._match_model_name("DenseNet201", model_name):
                self.model = DenseNet201(**args)
        elif re.match("Inception((V3)|(ResNetV2))", model_name):
            # InceptionV3, InceptionResNetV2

            if self._match_model_name("InceptionV3", model_name):
                self.model = InceptionV3(**args)
            elif self._match_model_name("InceptionResNetV2", model_name):
                self.model = InceptionResNetV2(**args)
        elif re.match("MobileNet(V[23])?((Small)|(Large))?", model_name):
            # MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large

            if self._match_model_name("MobileNetV2", model_name):
                self.model = MobileNetV2(**args)
            elif self._match_model_name("MobileNetV3Small", model_name):
                self.model = MobileNetV3Small(**args)
            elif self._match_model_name("MobileNetV3Large", model_name):
                self.model = MobileNetV3Large(**args)
            elif self._match_model_name("MobileNet", model_name):
                self.model = MobileNet(**args)
        elif re.match("ResNet((50)|(101)|(152))(V2)?", model_name):
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
            self.model.trainable = False
            self.__add_new_model_head()

        return self.model

    def predict(self, file_name_model: str, key: str = "test"):
        # load model
        model = load_model(self.paths["models"].joinpath(file_name_model))

        if model.input.shape[-1] == 1:
            self.color_mode = "grayscale"
        elif model.input.shape[-1] == 3:
            self.color_mode = "rgb"
        else:
            raise ValueError

        gen = self.get_data_generator(key, shuffle=False)
        y_prd_softmax = model.predict(x=gen,
                                      batch_size=self.determine_batch_size(key),
                                      verbose=self.verbose
                                      )

        y_prd = np.argmax(y_prd_softmax, axis=1)
        y_act = gen.classes
        y_false = np.nonzero(y_act != y_prd)[0]
        y_names = list(gen.class_indices.keys())

        # evaluate model on test set
        score = model.evaluate(gen, verbose=self.verbose)

        return gen, \
               pd.Series(y_prd, name="y_prd"), \
               pd.Series(y_act, name="y_act"), \
               pd.Series(y_false, name="y_false"), \
               pd.Series(y_names, name="y_names"), \
               score


if __name__ == "__main__":
    path_to_data_folder = pl.Path("Data")
    path_to_folder_save_models = pl.Path("Models")

    # TODO: more models: https://tfhub.dev/google/imagenet/inception_v1/classification/4
    models_to_analyze = ["MobileNetV2-pretrained-finetune"]

    train = TrainModelsImageClassification(path_to_data_folder,
                                           epochs=2,
                                           verbose=True,
                                           path_to_save_models=path_to_folder_save_models)
    for mdl in models_to_analyze:
        train.analyze(mdl)
    print('done.')
