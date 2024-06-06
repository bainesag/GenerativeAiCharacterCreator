@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--api
start "webServer" cmd /k "AiCharacterCreator\stable-diffusion-webui-master\venv\scripts\activate && python AiCharacterCreator/CharacterCreatorWebserver.py"