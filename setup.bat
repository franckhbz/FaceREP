@echo off

:: Crear el entorno virtual
python -m venv venv

:: Activar el entorno virtual
call .\venv\Scripts\activate

:: Actualizar pip
python -m pip install --upgrade pip

:: Instalar dependencias
pip install -r requirements.txt

:: Confirmar la instalación
pip list

:: Mantener la ventana abierta
pause
