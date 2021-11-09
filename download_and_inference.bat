@echo off

echo update steel lme data
start /w update_steel_lme.bat

echo download data from yahoo finance ...
start /w yahoo_downloading.bat

echo merging all data ... 
start /w preprocessing.bat

echo predicting scrap price (at next 3 month)... (machine learning)
start /w ml_inference.bat

echo predicting scrap price (at next 3 month) ... (deep learning)
start /w inference_dl_3month.bat

echo predicting scrap price (at next 1 week) ... (deep learning)
start /w inference_dl_1week.bat

pause