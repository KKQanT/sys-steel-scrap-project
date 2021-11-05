@echo off
echo update steel lme data
start /w update_steel_lme.bat
echo download data from yahoo finance ...
start /w yahoo_downloading.bat
echo merging all data ... 
start /w preprocessing.bat
pause