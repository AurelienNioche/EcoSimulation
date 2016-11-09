#!/usr/bin/env bash

echo "Switch to python 2.7.12"
pyenv local 2.7.12

echo "Call 'avakas_launcher.py'"
python avakas/avakas_launcher.py

echo "Come back to python 3.5.2"
pyenv local 3.5.2