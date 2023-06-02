import os
import sys
import numpy as np
import time
import shutil
sys.path.append(os.path.join("..", "..", "libs"))
from Simulators import H2CaveSimulator
from Utils import *

def write_settings(settings):
	# Define time levels
	n_steps = 100
	t_f = 2716*hour
	settings["Time"]["timeList"] = list(np.linspace(0, t_f, n_steps))

	# Define boundary conditions
	for u_i in settings["BoundaryConditions"].keys():
		for boundary_name in settings["BoundaryConditions"][u_i]:
			settings["BoundaryConditions"][u_i][boundary_name]["value"] = list(np.repeat(0.0, n_steps))
	settings["BoundaryConditions"]["u_z"]["TOP"]["value"] = list(np.repeat(-18.7*MPa, n_steps))

	# Create a name using the elements of the model
	name = settings["Model"][0]
	for elem in settings["Model"][1:]:
		name += f"_{elem}"
	settings["Paths"]["Output"] = os.path.join(*settings["Paths"]["Output"].split("/"), name)
	print(f"Output folder: {settings["Paths"]["Output"]}")

	# # Dump to file
	return settings

def main():
	# Read settings
	settings = read_json("settings.json")

	# Write settings
	settings = write_settings(settings)

	# Build simulation and run
	H2CaveSimulator(settings)


if __name__ == "__main__":
	main()