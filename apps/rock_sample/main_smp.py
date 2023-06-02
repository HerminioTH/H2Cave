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

def write_input_bc(input_bc):
	# Define time levels
	n_steps = 100
	time = np.linspace(0.0, 2716*hour, n_steps)
	input_bc["Time"]["timeList"] = list(time)

	# Define stress tensors
	input_bc["sigma_xx"] = list(np.repeat(0.0, n_steps))
	input_bc["sigma_xy"] = list(np.repeat(0.0, n_steps))
	input_bc["sigma_xz"] = list(np.repeat(0.0, n_steps))
	input_bc["sigma_yy"] = list(np.repeat(0.0, n_steps))
	input_bc["sigma_yz"] = list(np.repeat(0.0, n_steps))
	input_bc["sigma_zz"] = list(np.repeat(-18.7*MPa, n_steps))
	return input_bc

def write_output_folder(input_model):
	'''
		This function is totally optional, meaning that it can be removed without compromising the simulation.
		Its only purpose is to write the name of the output folder by combining the name of the elements composing the model.
	'''
	# Define output folder
	name = input_model["Model"][0]
	for elem in input_model["Model"][1:]:
		name += f"_{elem}"
	input_model["Paths"]["Output"] = os.path.join(*input_model["Paths"]["Output"].split("/"), "smp", name)
	output_folder = os.path.join(input_model["Paths"]["Output"])
	print(f"Output folder: {output_folder}")
	return input_model

def main():
	# Read input_model
	input_model = read_json("input_model.json")
	input_model = write_output_folder(input_model)

	# Read input_bc
	input_bc = read_json("input_bc_smp.json")
	input_bc = write_input_bc(input_bc)

	# Run simulator
	SmpSimulator(input_model, input_bc)


if __name__ == '__main__':
	main()
