#!/bin/bash

modG=("firstTry.npy" "dropoutTry.npy" "alldropTry.npy")      	       		# Nome del file di output previsioni
titM=("firstTry" "dropoutTry" "alldropTry")      	       		        # Nome del file di output previsioni

#-----------------------------------------------------------------------#
#       Ciclo per modificare parametri esecuzione script confM.py       #
#-----------------------------------------------------------------------#
for j in "${!modG[@]}"; do
        python3 confM.py ${modG[$j]}  ${titM[$j]}
done

echo "Stampa delle matrici di confusione terminata!"
