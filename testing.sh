#!/bin/bash

python3 test.py 'rmse'
python3 test.py 'mse'
python3 test.py 'mae'
python3 test.py 'mre'
python3 test.py 'gap'

./makeMovie.sh
