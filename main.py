from API_Experiments import tools as API_Exp
from API_Experiments.api import Lab
from torchvision.datasets import MNIST
from Experimentos.descenso_gradiente import descenso_gradiente
import numpy as np


def initialize_dataset() -> dict:
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

    # Debe retornar con la siguiente estructura

    return {"train": (df_trainX, df_trainY),
            "valid": (),
            "test": (df_testX, df_testY)
            }


conf_dataset = {"size": 100,
                "class_selected": [0, 1],
                "comparador": [(0, 1)],
                "fun_perturbation": None
                }

def define_experiment():
    pass

inputs = {"EXPERIMENT": {"ALGORITHMS": [("Descenso del Gradiente", {"alpha": 0.1,
                                                                    "beta": 0.2}
                                         )],
                         },
          }


################################# NO TOCAR ##################################
################## A MENOS QUE SEPAS LO QUE ESTAS HACIENDO ##################

def main():
    # PREPARANDO LOS EXPERIMENTOS
    # Creamos inicializamos el objeto Lab_Experiment
    lab: Lab = Lab()

    # Preparamos nuestro dataset que usaremos en nuestras mediciones
    lab.set_dataset(initialize_dataset())

    # Preparamos los inputs
    lab.set_inputs(inputs)

    # Preparamos el output
    lab.prepared_outputs()

    # CORRIENDO LOS EXPERIMENTOS
    lab.run_experiments()

    # Experimento Terminado
    # Revisar ".csv" correspondientes

    # Creamos o cargamos nuestro dataset de datos donde
    # se guardaran los datos obtenidos del experimento,
    # puede ser uno nuevo o añadir los datos a uno anterior
    new_dataset_metadata: bool = False
    API_Exp.initial_df_metadata(new=new_dataset_metadata)

    # Filtrar y perturbar el dataset de las features (df_trainX, df_testX)
    # según lo estimado en el experimento
    # Filtrar por tamaño y clases, y aplicar función de perturbación
    size_of_dataset: int = 100
    class_selected: list[int] = [0, 1]
    comparator: list[tuple[int, int]] = [(0, 1)]
    fun_perturbation: callable(float) = lambda t: t + 1

    API_Exp.set_dataset_by_experiment(size=size_of_dataset,
                                      class_filter=class_selected,
                                      class_to_compare=comparator,
                                      fun_per=fun_perturbation)

    # Listos los datos a utilizar para el experimento y tenemos donde guardarlo
    # procedemos con el experimento
    API_Exp.run_experiments()


if __name__ == "__main__":
    main()
