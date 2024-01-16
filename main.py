import API_experiment as API_Exp

# Creamos o cargamos nuestro dataset de datos donde
# se guardaran los datos obtenidos del experimento,
# puede ser uno nuevo o añadir los datos a uno anterior

new_dataset_metadata: bool = False
API_Exp.initial_df_metadata(new=new_dataset_metadata)

# Obtenemos nuestro dataset que usaremos en nuestras mediciones
API_Exp.initial_dataset()

# Filtrar y perturbar el dataset de las features (df_trainX, df_testX)
# según lo estimado en el experimento
# Filtrar por tamaño, clase y aplicar función de perturbación
size_of_dataset: int = 100
class_selected: list[int] = [0, 1]
comparator: list[tuple[int, int]] = [(0, 1)]
fun_perturbation: callable(float) = lambda t: t + 1

API_Exp.set_dataset_by_experiment(size=size_of_dataset,
                                  class_filter=class_selected,
                                  fun_per=fun_perturbation)

# Listos los datos a utilizar para el experimento y tenemos donde guardarlo
# procedemos con el experimento


# Experimento Terminado


"""clean_dataset: bool = True
perturbe_dataset: bool = True




if clean_dataset:
    API_Exp.initial_clean_df(comb_num)

if perturbe_dataset:
    API_Exp.perturbe_df()

API_Exp.initial_dataset()"""
