import numpy as np
import pandas as pd
import json
import os
import sys
sys.path.append(os.path.join("..", "..", "..", "MechanicsAnalytic", "libs"))
from AnalyticSolutions import *

def read_json(file_name):
	with open(file_name, "r") as j_file:
		data = json.load(j_file)
	return data

def save_json(data, file_name):
	with open(file_name, "w") as f:
	    json.dump(data, f, indent=4)

def read_settings():
	# Read settings
	settings_original = read_json("settings.json")

	# print(settings_original["Time"])

	n = len(settings_original["Time"]["timeList"])
	settings = {
		"viscoplastic": settings_original["Viscoplastic"],
		"time": settings_original["Time"]["timeList"],
		"sigma_xx": settings_original["BoundaryConditions"]["u_x"]["OUTSIDE"]["value"],
		"sigma_yy": settings_original["BoundaryConditions"]["u_x"]["OUTSIDE"]["value"],
		"sigma_zz": settings_original["BoundaryConditions"]["u_z"]["TOP"]["value"],
		"sigma_xy": [0.0 for i in range(n)],
		"sigma_yz": [0.0 for i in range(n)],
		"sigma_xz": [0.0 for i in range(n)]
	}

	settings["viscoplastic"]["eta"] = settings["viscoplastic"]["eta_1"]
	settings["viscoplastic"]["m"] = settings["viscoplastic"]["m_v"]

	# print()
	# print(settings)

	return settings

def main():
	# Deine output folder
	output_folder = os.path.join("output", "test_0")

	# Write stresses, if necessary
	settings = read_settings()

	# Initialize models
	model_vp = ViscoPlastic_Desai(settings)

	# Compute strains
	model_vp.compute_strains()



if __name__ == '__main__':
	main()
