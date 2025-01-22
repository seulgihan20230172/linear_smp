#!/bin/bash

# Python 스크립트 실행
python3 linear_csv.py

# 결과 파일 분석
echo "Analyzing results..."
BEST_ROW=$(awk -F',' 'NR==1 { for (i=1; i<=NF; i++) {header[i]=$i} } 
NR>1 && ($NF < min || min=="") {min=$NF; row=$0} 
END {print row}' lstm_results.csv)

echo "Best configuration:"
echo "$BEST_ROW"
