#!/bin/bash

modG=("new_L1.npy")      	# Nome del file di output previsioni
titM=("new_L1")	               		# Nome output matrice di confusione

#-----------------------------------------------------------------------#
#       Ciclo per modificare parametri esecuzione script confM.py       #
#-----------------------------------------------------------------------#
for j in "${!modG[@]}"; do
        python3 confM.py ${modG[$j]}  ${titM[$j]}
done

echo "Stampa delle matrici di confusione terminata!"
