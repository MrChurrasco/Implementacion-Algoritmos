from API_Experiments import tools as API_Exp
from Experimentos.descenso_gradiente import descenso_gradiente
import funciones_costo as fc
import pandas as pd


def test_input_function_to_dataframe():
    one_element = ("square_error", fc.square_loss)

    dataframe1, list_fun1 = API_Exp.parse_input_function_to_dataframe(one_element)
    assert list_fun1 == [fc.square_loss]
    assert dataframe1.columns.to_list() == ["name_function", "latex"]
    assert dataframe1.name_function.to_list() == ["square_loss"]

    list_element = [("square_loss", fc.square_loss),
                    ("hinge_loss", fc.hinge_loss),
                    ("exp_loss", fc.exp_loss)
                    ]
    dataframe2, list_fun2 = API_Exp.parse_input_function_to_dataframe(list_element)
    assert list_fun2 == [fc.square_loss, fc.hinge_loss, fc.exp_loss]
    assert dataframe2.columns.to_list() == ["name_function", "latex"]
    assert dataframe2.name_function.to_list() == ["square_loss", "hinge_loss", "exp_loss"]


def test_dict_descent_methods_input_to_dataframe():
    # TEST ONE METHOD
    # WITH ONE INPUT

    one_method_one_input = {"descent_gradient": (descenso_gradiente, {"alpha": 1,
                                                                      "beta": 0.5
                                                                      }
                                                 )
                            }
    dataframe1, dict_dataframe_methods1, list_methods1 = API_Exp.parse_dict_descent_methods_input_to_dataframe(
        one_method_one_input)

    assert dataframe1 == pd.DataFrame({"name_methods": ["descent_gradient"],
                                       "method_idx": [0]
                                       })
    assert dict_dataframe_methods1["descent_gradient"] == pd.DataFrame({"alpha": [1],
                                                                        "beta": [0.5]
                                                                        })
    assert list_methods1 == [descenso_gradiente]

    # WITH MANY INPUTS

    one_method_many_input = {"descent_gradient": (descenso_gradiente, {"alpha": [1, 0.1, 2],
                                                                       "beta": [0.5, 1]
                                                                       }
                                                  )
                             }

    dataframe2, dict_dataframe_methods2, list_methods2 = API_Exp.parse_dict_descent_methods_input_to_dataframe(
        one_method_many_input)

    assert dataframe2 == pd.DataFrame({"name_methods": ["descent_gradient", "descent_gradient", "descent_gradient",
                                                        "descent_gradient", "descent_gradient", "descent_gradient"],
                                       "method_idx": [0, 1, 2, 3, 4, 5]
                                       })
    assert dict_dataframe_methods2["descent_gradient"] == pd.DataFrame({"alpha": [1, 0.1, 2, 1, 0.1, 2],
                                                                        "beta": [0.5, 0.5, 0.5, 1, 1, 1]
                                                                        })
    assert list_methods2 == [descenso_gradiente]

    # TEST MANY METHODS
    # WITH ONE INPUT

    def methodA(x, y):
        return 0

    def methodB(x, y, z):
        return 1

    many_method_one_input = {"descent_gradient": (descenso_gradiente, {"alpha": [1],
                                                                       "beta": [0.5]
                                                                       }
                                                  ),
                             "methodA": (methodA, {"x": [1],
                                                   "y": [4]
                                                   }
                                         ),
                             "methodB": (methodB, {"x": [2],
                                                   "y": [3],
                                                   "z": [4]
                                                   })
                             }

    dataframe3, dict_dataframe_methods3, list_methods3 = API_Exp.parse_dict_descent_methods_input_to_dataframe(
        many_method_one_input)

    assert dataframe3 == pd.DataFrame({"name_methods": ["descent_gradient", "methodA", "methodB"],
                                       "method_idx": [0, 0, 0]
                                       })
    assert dict_dataframe_methods3["descent_gradient"] == pd.DataFrame({"alpha": [1, ],
                                                                        "beta": [0.5]
                                                                        })
    assert dict_dataframe_methods3["methodA"] == pd.DataFrame({"x": [1],
                                                               "y": [4]
                                                               })
    assert dict_dataframe_methods3["methodB"] == pd.DataFrame({"x": [2],
                                                               "y": [3],
                                                               "z": [4]
                                                               })
    assert list_methods3 == [descenso_gradiente, methodA, methodB]

    # WITH MANY INPUTS
    many_method_many_input = {"descent_gradient": (descenso_gradiente, {"alpha": [1, 2],
                                                                        "beta": [0.5, 1]
                                                                        }
                                                   ),
                              "methodA": (methodA, {"x": [1,2],
                                                    "y": [4,5]
                                                    }
                                          ),
                              "methodB": (methodB, {"x": [2,1],
                                                    "y": [3,0],
                                                    "z": [4,5]
                                                    })
                              }

    dataframe4, dict_dataframe_methods4, list_methods4 = API_Exp.parse_dict_descent_methods_input_to_dataframe(
        many_method_many_input)

    assert dataframe4 == pd.DataFrame({"name_methods": ["descent_gradient", "methodA", "methodB"],
                                       "method_idx": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7]
                                       })
    assert dict_dataframe_methods4["descent_gradient"] == pd.DataFrame({"alpha": [1, 2, 1, 2],
                                                                        "beta": [0.5, 0.5, 1, 1]
                                                                        })
    assert dict_dataframe_methods4["methodA"] == pd.DataFrame({"x": [1, 2, 1, 2],
                                                               "y": [4, 4, 5, 5]
                                                               })
    assert dict_dataframe_methods4["methodB"] == pd.DataFrame({"x": [2, 1, 2, 1, 2, 1, 2, 1],
                                                               "y": [3, 3, 0, 0, 3, 3, 0, 0],
                                                               "z": [4, 4, 4, 4, 5, 5, 5, 5]
                                                               })
    assert list_methods4 == [descenso_gradiente, methodA, methodB]

def test_errors_set_dataset_by_experiment():
    # Errors in size
    size_error = 0
    API_Exp.set_dataset_by_experiment()


def test_add_new_row_to_DataFrame():
    pass


"""
@pytest.mark.parametrize(
    "t_test, expected",
    [(("Sum1", lambda x: x + 1),
      pd.DataFrame({"name": ["Sum1"],
                    "latex": []
                    })
      )]
)
def perturbation_function_dict_to_dataframes():
    pass
"""
