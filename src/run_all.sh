#!/bin/bash
# Start both networks in the background, then open the dashboard.
cd "$(dirname "$0")/.."
mkdir -p logs
[ -f data/corpus.npy ] || python3 src/build_corpus.py
nohup python3 src/train_a.py > logs/a.log 2>&1 &
echo "A started (pid $!)  -> logs/a.log"
sleep 20                                   # let A write its first checkpoint
nohup python3 src/train_b.py > logs/b.log 2>&1 &
echo "B started (pid $!)  -> logs/b.log"
sleep 2
python3 src/dashboard.py
