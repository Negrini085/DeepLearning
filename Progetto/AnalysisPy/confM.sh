#!/bin/bash

modG=("emo_v1.npy" "emo_v2.npy")      	                # Nome del file di output previsioni
titM=("emo_v1" "emo_v2")      	       		        # Nome output matrice di confusione

#-----------------------------------------------------------------------#
#       Ciclo per modificare parametri esecuzione script confM.py       #
#-----------------------------------------------------------------------#
for j in "${!modG[@]}"; do
        python3 confM.py ${modG[$j]}  ${titM[$j]}
done

echo "Stampa delle matrici di confusione terminata!"
