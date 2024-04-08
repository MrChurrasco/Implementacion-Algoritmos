import numpy as np
from typing import Any


class Algorithm:
    def __init__(self, name, inputs, outputs):
        self.__name: str = name
        self.__inputs: dict[str, type] = inputs
        self.__outputs: dict[str, type] = outputs
        self.__lab: Any = None

    def get_name(self) -> str:
        return self.__name

    def get_inputs(self) -> dict[str, type]:
        return self.__inputs

    def get_outputs(self) -> dict[str, type]:
        return self.__outputs

    def set_lab(self, lab_experiment) -> None:
        self.__lab = lab_experiment

    def save_output(self, output: dict[str, Any]) -> None:
        self.__lab.save_output(output, foreach=True)

    def run_algorithm(self, **args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def consistency_test(self):
        if type(self.__name) is not str:
            raise TypeError(f'{self.__name} no es un string.')

        def input_output_test(x: Any):
            if type(x) is not dict:
                raise TypeError(f'{x} no es un diccionario.')
            for key, value in x.items():
                if type(key) is not str:
                    raise TypeError(f'{key} no es un string.')
                if type(value) is not type:
                    raise TypeError(f'{value} no es un tipo.')

        input_output_test(self.__inputs)
        input_output_test(self.__outputs)

        if issubclass(type(self.__lab), Lab):
            raise TypeError(f'{self.__lab} no es una clase hija de Lab.')


class Lab:
    def __init__(self):
        self.__dict_algorithms: dict = {}

    def set_dataset(self, data: np.ndarray,
                    data_train=None,
                    data_valid=None,
                    data_test=None,
                    complete_datasets=False) -> None:
        pass

    def set_inputs(self, input_data: dict | None) -> None:
        pass

    def set_algorithm(self, algorithms: dict) -> None:
        if type(algorithms) is not dict:
            raise TypeError(f'{algorithms} no es un diccionario.')
        for key, value in algorithms.items():
            if type(key) is not str:
                raise TypeError(f'{key} no es un string.')
            if type(value.__class__.__base__) is not Algorithm:
                raise TypeError(f'{value} no es un clase hija de Algorithm.')
        self.__dict_algorithms = algorithms

    def save_output(self, output: dict[str, Any], foreach=False) -> None:
        pass
    def run_experiments(self, experiment: callable):
        pass





