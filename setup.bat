@echo off
echo creating necessary folder ...
start /w create_necessary_folder.bat
echo necessary folder created
echo install necessary python packages ...
start /w install_packages.bat
echo finished installation process
echo writing model config...
start /w reset_config.bat
pause