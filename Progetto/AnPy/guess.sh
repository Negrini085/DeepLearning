#!/bin/bash

modN=("line3_v2.keras")      	             # Nome del file di input modello
modG=("line3_v2.npy")      	                     # Nome del file di output previsioni

#-----------------------------------------------------------------------#
#       Ciclo per modificare parametri esecuzione script guess.py       #
#-----------------------------------------------------------------------#
for j in "${!modN[@]}"; do
        python3 guess.py ${modN[$j]} ${modG[$j]}
done

echo "Determinazione delle guess terminata!"
