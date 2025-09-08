#!/bin/bash

modG=("try1.npy" "try2.npy" "try3.npy" "try4.npy" "try5.npy")      	                # Nome del file di output previsioni
titM=("try1" "try2" "try3" "try4" "try5")      	       		                	# Nome output matrice di confusione

#-----------------------------------------------------------------------#
#       Ciclo per modificare parametri esecuzione script confM.py       #
#-----------------------------------------------------------------------#
for j in "${!modG[@]}"; do
        python3 confM.py ${modG[$j]}  ${titM[$j]}
done

echo "Stampa delle matrici di confusione terminata!"
