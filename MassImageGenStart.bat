@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--api
start "MassImageGen" cmd /k "AiCharacterCreator\stable-diffusion-webui-master\venv\scripts\activate && python AiCharacterCreator/MassImageGen.py"