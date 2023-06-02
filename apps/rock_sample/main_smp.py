import numpy as np
import pandas as pd
import json
import os
import stat
import sys
sys.path.append(os.path.join("..", "..", "libs"))
# from RockSampleSolutions import *
from Simulators import SmpSimulator
from Utils import read_json, save_json, hour, MPa
import time

def main():
	# Read input_model
	input_model = read_json("input_model.json")

	# Read input_bc
	input_bc = read_json("input_bc_smp.json")

	# Run simulator
	SmpSimulator(input_model, input_bc)


if __name__ == '__main__':
	main()
