#!/bin/bash

pip uninstall TCAT2Phase
python3 setup.py bdist_wheel
pip install dist/TCAT2Phase-0.1-py3-none-any.whl


