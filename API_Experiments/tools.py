import numpy as np
from torchvision.datasets import MNIST
import pandas as pd

# Variables globales
"""
# Dataframes del entrenamiento
df_trainX: np.ndarray
df_trainY: np.ndarray
df_testX: np.ndarray
df_testY: np.ndarray

# DataFrames que guardarán los datos obtenidos en el entrenamiento
df_main: pd.DataFrame
df_comb: pd.DataFrame
df_iter: pd.DataFrame

# Comparaciones que se harán en el experimento
list_to_compare: set[tuple[int, int]]
"""

# Los nombres de los archivos .csv
name_df_main: str = "DF_ALGORITHMS.csv"
name_df_comb: str = "DF_DATA_COMBINATION.csv"
name_df_iter: str = "DF_ITERATIONS.csv"


# Funciones de la API
def initialize_dataset_training() -> None:
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


def set_dataset_by_experiment(size: int = 0,
                              class_filter: list[int] = None,
                              class_to_compare: list[tuple[int, int]] = None,
                              fun_per: callable(float) = None,
                              seed: int = 1234):
    global df_trainX, df_trainY, df_testX, df_testY, list_to_compare

    # Error que pueden suceder
    if type(size) is not int and size <= 0:
        raise ValueError("size tiene que ser entero positivo y mayor que 0 o None.")

    if type(seed) is not int and seed > 0:
        raise ValueError("seed tiene que ser entero positivo y mayor que 0.")

    if type(fun_per) is not callable and fun_per is not None:
        raise ValueError("fun_per tiene que ser una función.")

    if type(class_filter) is not None:
        if type(class_filter) is not list:
            raise ValueError("class_filter debe ser una lista de enteros.")
        if len(class_filter) < 1:
            raise ValueError("class_filter debe ser una lista que contenga más de una clase.")
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

    # Filtrado
    # Obtenemos las clases que deseamos quedarnos
    filtro_clases: set[int] = set()
    if class_to_compare is not None:
        aux_1, aux_2 = zip(*class_to_compare)
        filtro_clases = set(aux_1) | set(aux_2) | filtro_clases
        del aux_1, aux_2
    if class_filter is not None:
        filtro_clases = set(class_filter) | filtro_clases

    # Ya obtenida las clases que vamos a necesitar filtramos el dataset
    df_trainX = df_trainX[np.in1d(df_trainY, list(filtro_clases))]
    df_trainY = df_trainY[np.in1d(df_trainY, list(filtro_clases))]
    df_testX = df_testX[np.in1d(df_testY, list(filtro_clases))]
    df_testY = df_testY[np.in1d(df_testY, list(filtro_clases))]

    # Comparaciones que se realizarán en el experimento
    # Guardamos las comparaciones que se van a realizar en el experimento
    if class_to_compare:
        list_to_compare = set([(x, y) if x < y else (y, x) for x, y in class_to_compare])
    elif class_filter:
        list_to_compare = set()
        # cross class_filter consigo misma
        for x in class_filter:
            for y in class_filter:
                if x < y:
                    list_to_compare.add((x, y))
    else:
        list_to_compare = set()
        aux = set(df_testY.tolist())
        for x in aux:
            for y in aux:
                if x < y:
                    list_to_compare.add((x, y))
        del aux

    # Aplicando la función de perturbación a las features
    if fun_per:
        # Vectorizamos la función de perturbación
        v_fun_per = np.vectorize(fun_per)
        # Aplicamos en las features
        df_trainX = v_fun_per(df_trainX)
        df_testX = v_fun_per(df_testX)

    # Recortamos el dataset según el tamaño pedido
    if size > 0:
        df_trainX = np.random.choice(df_trainX, size, replace=False)
        df_testX = np.random.choice(df_testX, size, replace=False)

    # Todo esta listo para el loop del experimento


def run_experiments():
    pass


def parse_dict_descent_methods_input_to_dataframe(one_method_one_input):
    return None


def parse_input_function_to_dataframe(one_element):
    return None

    # Filtrar y perturbar el dataset de las features (df_trainX, df_testX)
    # según lo estimado en el experimento
    # Filtrar por tamaño y clases, y aplicar función de perturbación
    size_of_dataset: int = 100
    class_selected: list[int] = [0, 1]
    comparator: list[tuple[int, int]] = [(0, 1)]
    fun_perturbation: callable(float) = lambda t: t + 1