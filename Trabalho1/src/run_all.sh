#!/bin/sh
# Executa todos os experimentos em sequência (retomável: dobras prontas são puladas).
cd "$(dirname "$0")"
python3 run_main.py RF > ../results/log_RF.txt 2>&1
python3 run_main.py LR > ../results/log_LR.txt 2>&1
python3 run_rakel_k.py RF > ../results/log_rakel_RF.txt 2>&1
python3 run_lp_decoding.py > ../results/log_lp_decoding.txt 2>&1
python3 analyze.py > ../results/analyze_out.txt 2>&1
