#!/bin/bash

modG=("try1W.npy")      	# Nome del file di output previsioni
titM=("try1W")	               		# Nome output matrice di confusione

#-----------------------------------------------------------------------#
#       Ciclo per modificare parametri esecuzione script confM.py       #
#-----------------------------------------------------------------------#
for j in "${!modG[@]}"; do
        python3 confM.py ${modG[$j]}  ${titM[$j]}
done

echo "Stampa delle matrici di confusione terminata!"
