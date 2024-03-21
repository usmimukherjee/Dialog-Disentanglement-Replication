#!/bin/bash

echo "Setting DIS_HOME to the local repository base path."
DIS_HOME="D:\BugMentor\Dialog-Disentanglement-Replication\\"
input_file="${DIS_HOME}/data/train/2004-12-25.train-c.ascii.txt"
tmpfile=todo.$RANDOM

# No need to create a tmpfile from stdin. We will use the existing input_file instead.
# Copy the input file to a temporary file with .ascii.txt extension
cp "${input_file}" "${tmpfile}.ascii.txt"

echo "Running preprocessing tokenization script."
python "${DIS_HOME}/tools/preprocessing/dstc8-tokenise.py" --vocab "${DIS_HOME}/data/vocab.txt" --output-suffix .tok "${tmpfile}.ascii.txt"

echo "Moving tokenized file to a new location."
mv "${tmpfile}.ascii.txt.tok" "${tmpfile}.tok.txt"

echo "Running the main disentangle script."
python -u "${DIS_HOME}/src/dialog_disentanglement.py" \
  "${tmpfile}" \
  --train ../data/train/*annotation.txt \
  --dev ../data/dev/*annotation.txt \
  --hidden 512 \
  --layers 2 \
  --nonlin softsign \
  --word-vectors ../data/glove-ubuntu.txt \
  --epochs 20 \
  --dynet-autobatch \
  --drop 0 \
  --learning-rate 0.018804 \
  --learning-decay-rate 0.103 \
  --seed 10 \
  --clip 3.740 \
  --weight-decay 1e-07 \
  --opt sgd \
  > "${tmpfile}.out" 2>example-train.err

echo "Processing the output file."
cat "${tmpfile}.out" | grep -v '^[#]' | sed 's/.*[:]//'

echo "Script execution completed."
