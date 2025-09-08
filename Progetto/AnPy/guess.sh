#!/bin/bash

modN=("try1.keras" "try2.keras" "try3.keras" "try4.keras" "try5.keras")      	             # Nome del file di input modello
modG=("try1.npy" "try2.npy" "try3.npy" "try4.npy" "try5.npy")      	                     # Nome del file di output previsioni

#-----------------------------------------------------------------------#
#       Ciclo per modificare parametri esecuzione script guess.py       #
#-----------------------------------------------------------------------#
for j in "${!modN[@]}"; do
        python3 guess.py ${modN[$j]} ${modG[$j]}
done

echo "Determinazione delle guess terminata!"
