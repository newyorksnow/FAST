# FAST
All of the code used to carry out the experiments in the paper. 
The webiste "Meteorismo": https://sites.google.com/view/meteorismo-safety-forecast/home

Guide:

cleandata - preparing MSD by combining datasets

lstm-multidata - LSTM model trained on MSD (with some preprocessing)

lstm-gtd - LSTM model trained on GTD (with some preprocessing)

MSD_GTD_comparison - code used to evaluate differences in both datasets

p-tests - code that carries out a paired t-test on the results from lstm-multidata and lstm-gtd
