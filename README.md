# Implementacion-Algoritmos

 Implementación de algoritmos de optimización y diferentes funciones de convexas y no convexas.

## ¿Qué es necesario hacer antes de comenzar a utilizar o modificar el proyecto?
El proyecto esta en [**Python 3.11**](https://www.python.org/downloads/release/python-3117/).

Para trabajar en este proyecto se debe instalar la versión antes dicha de Python y crear un entorno virtual. Para ello 
escriba el comando:

```{python}
python -m venv venv
```

Para instalar las librerías antes que se utilizan en este proyecto solo debe entrar con el comando:

```{python}
venv\Scripts\activate
```

Ya dentro del entorno virtual en el cmd o powershell, escriba:

```{python}
pip install -r requirements.txt
```

Una vez que termine la instalación, esta listo para ejecutar el código.


**Este proceso solo se hace una vez.**

## ¿Cómo utilizar el testeo?

Este proyecto esta testeado con [Pytest](https://docs.pytest.org/en/7.4.x/), de mode que si realiza algún cambio es 
necesario que realice el siguiente comando en el entorno virtual:
```{python}
pytest
```
Este comando dará un panorama completo del testeo. Si se desea mayor detalle utilice el siguiente comando:
```{python}
pytest -v
```

Los errores los arrojará con un color rojizo.

