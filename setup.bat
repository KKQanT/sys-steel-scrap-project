@echo off
echo creating necessary folder ...
start /w create_necessary_folder.bat
echo necessary folder created
echo install necessary python packages ...
start /w install_venv.bat
echo finished installation process
pause