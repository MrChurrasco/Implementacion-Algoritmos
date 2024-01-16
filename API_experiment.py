import numpy as np
from torchvision.datasets import MNIST
from descenso_gradiente import descenso_gradiente
from time import perf_counter_ns
import pandas as pd

# Variables globales

# Dataframes del entrenamiento
df_trainX: np.ndarray
df_trainY: np.ndarray
df_testX: np.ndarray
df_testY: np.ndarray

x_train: np.ndarray
y_train: np.ndarray
x_test: np.ndarray
y_test: np.ndarray

# DataFrames que guardarán los datos obtenidos en el entrenamiento
df_main: pd.DataFrame
df_comb: pd.DataFrame
df_iter: pd.DataFrame

# Los nombres de los archivos .csv
name_df_main: str = "DF_ALGORITHMS.csv"
name_df_comb: str = "DF_DATA_COMBINATION.csv"
name_df_iter: str = "DF_ITERATIONS.csv"

# Comparaciones que se harán en el experimento
list_to_compare: set[tuple[int, int]]


# Funciones de la API
def initial_dataset() -> None:
    global df_trainX, df_trainY, df_testX, df_testY

    df_train: list = list(zip(*MNIST(root="",
                                     train=True,
                                     download=True,
                                     transform=lambda t: np.asarray(t, dtype=np.float32))))

    df_test: list = list(zip(*MNIST(root="",
                                    train=False,
                                    download=True,
                                    transform=lambda t: np.asarray(t, dtype=np.float32))))

    # Separamos las imagenes (x) de los labels (y)
    df_trainX: np.ndarray = np.asarray(df_train[0])
    df_trainY: np.ndarray = np.asarray(df_train[1])

    df_testX: np.ndarray = np.asarray(df_test[0])
    df_testY: np.ndarray = np.asarray(df_test[1])

    del df_train, df_test


def initial_df_metadata(new: bool = False) -> None:
    global df_main, df_comb, df_iter
    global name_df_main, name_df_comb, name_df_iter

    if new is not bool:
        raise TypeError("Argumento 'new' debe ser True o False")

    if new:

        # Dataframe que contendrá los datos de cada prueba de combinación
        columns_df_main: list = ["algorithm", "time", "num_iter", "value_grad", "data_index"]

        df_main = pd.DataFrame(columns=columns_df_main)
        df_main.to_csv(name_df_main)

        # Dataframe que contendrá los datos de cada iteración
        columns_df_iter: list = ["main_index", "time_iter", "value_grad_iter", "const_iter"]

        df_main = pd.DataFrame(columns=columns_df_iter)
        df_main.to_csv(name_df_iter)

        # Todas las comparaciones entre las clases
        comb_num = []
        for x in range(9):
            for y in range(9):
                if x < y:
                    comb_num.append((x, y))

        df_comb = pd.DataFrame(comb_num, columns=["number_1", "number_2"], dtype=np.int8)
        df_comb.to_csv(name_df_comb)

    else:
        try:
            df_comb = pd.read_csv(name_df_comb, index_col="Unnamed: 0")
            df_main = pd.read_csv(name_df_main, index_col="Unnamed: 0")
            df_iter = pd.read_csv(name_df_iter, index_col="Unnamed: 0")
        except FileNotFoundError:
            raise FileNotFoundError(f"Los archivos {name_df_comb}, {name_df_main} y {name_df_iter} no existen.\n",
                                    f"Marque new=True en caso que no existan estos archivos.")


def set_dataset_by_experiment(size: int,
                              class_filter: list[int] = None,
                              class_to_compare: list[tuple[int, int]] = None,
                              fun_per: callable(float) = None,
                              seed: int = 1234):
    global df_trainX, df_trainY, df_testX, df_testY, list_to_compare

    # Error que pueden suceder
    if type(size) is not int and size > 0:
        raise ValueError("size tiene que ser entero positivo y mayor que 0.")

    if type(seed) is not int and seed > 0:
        raise ValueError("seed tiene que ser entero positivo y mayor que 0.")

    if type(fun_per) is not callable and fun_per is not None:
        raise ValueError("fun_per tiene que ser una función.")

    if type(class_filter) is not None:
        if type(class_filter) is not list:
            raise ValueError("class_filter debe ser una lista de enteros.")
        for num in class_filter:
            if type(num) is not int and num >= 0:
                raise ValueError("class_filter tiene un elemento que no es un entero mayor igual que cero.")

    if type(class_to_compare) is not None:
        if type(class_to_compare) is not list:
            raise ValueError("class_to_compare debe ser una lista.")
        for num1, num2 in class_to_compare:
            if type(num1) is not int and num1 >= 0:
                raise ValueError("class_to_compare tiene un elemento que no es un entero mayor igual que cero.")
            if type(num2) is not int and num2 >= 0:
                raise ValueError("class_to_compare tiene un elemento que no es un entero mayor igual que cero.")

    for num1, num2 in class_to_compare:
        if num1 not in class_filter or num2 not in class_filter:
            raise ValueError(
                f"Los valores {num1} y/o {num2} que se desean comparar no se encuentra después de filtrar los datos.")

    # Código

    # Guardamos las comparaciones que se van a realizar en el experimento

    if class_to_compare:
        list_to_compare = set([(x, y) if x < y else (y, x) for x, y in class_to_compare])
    else:
        if class_filter:
            list_to_compare = set()
            # cross class_filter consigo misma
            for x in class_filter:
                for y in class_filter:
                    if x < y:
                        list_to_compare.add((x, y))
