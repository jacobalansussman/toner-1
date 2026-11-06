#!/usr/bin/env python3
"""Launcher for llama2.c train.py (patched copy in training/llama2c) under `python -I`, which drops the script dir from sys.path.
Usage: python -I -u pro_train.py [config.py] [--key=value ...]   (arguments go straight to llama2.c's configurator)
"""
import os, runpy, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import L2C  # noqa: E402  (also sets TONER_TRAIN for tinystories.py and the config)
CODE = L2C
# config files given as paths are made absolute before the chdir
sys.argv = [os.path.join(CODE, "train.py")] + [a if a.startswith("--") or not os.path.exists(a) else os.path.abspath(a)
                                                for a in sys.argv[1:]]
os.chdir(CODE)
sys.path.insert(0, CODE)
runpy.run_path(sys.argv[0], run_name="__main__")
