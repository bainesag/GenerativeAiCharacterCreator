@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--api
start AiCharacterCreator\stable-diffusion-webui-master\venv\scripts\activate & python CharacterCreatorWebserver.py
cd AiCharacterCreator\stable-diffusion-webui-master
call webui.bat