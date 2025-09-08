#!/bin/bash

modG=("line1_v1.npy" "line1_v2.npy" "line2_v1.npy" "line2_v2.npy")      	# Nome del file di output previsioni
titM=("line1_v1" "line1_v2" "line2_v1" "line2_v2")	               		# Nome output matrice di confusione

#-----------------------------------------------------------------------#
#       Ciclo per modificare parametri esecuzione script confM.py       #
#-----------------------------------------------------------------------#
for j in "${!modG[@]}"; do
        python3 confM.py ${modG[$j]}  ${titM[$j]}
done

echo "Stampa delle matrici di confusione terminata!"
