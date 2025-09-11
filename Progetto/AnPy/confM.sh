#!/bin/bash

modG=("line2_v2.npy" "line3_v2.npy")      	# Nome del file di output previsioni
titM=("line2" "line3")	               		# Nome output matrice di confusione

#-----------------------------------------------------------------------#
#       Ciclo per modificare parametri esecuzione script confM.py       #
#-----------------------------------------------------------------------#
for j in "${!modG[@]}"; do
        python3 confM.py ${modG[$j]}  ${titM[$j]}
done

echo "Stampa delle matrici di confusione terminata!"
