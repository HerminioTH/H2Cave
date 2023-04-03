import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import json
import os
import sys
sys.path.append(os.path.join("..", "..", "..", "MechanicsAnalytic", "libs"))
from AnalyticSolutions import *

sec = 1.
minute = 60*sec
hour = 60*minute

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
		"elasticity": settings_original["Elastic"],
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
	output_folder = os.path.join("output", "case_0", "ana")

	# Write stresses, if necessary
	settings = read_settings()

	# Initialize models
	model_e = Elastic(settings)
	model_vp = ViscoPlastic_Desai(settings)

	# Compute strains
	model_e.compute_strains()
	model_vp.compute_strains()

	# Compute total strain
	eps_tot = model_e.eps.copy()
	eps_tot += model_vp.eps

	# print()
	# print(settings["sigma_zz"])
	# print(settings["sigma_xx"])
	# print(settings["sigma_yy"])
	# print(model_e.eps[:,2,2])

	# Save results
	saver_eps_vp = TensorSaver(output_folder, "eps_vp")
	saver_eps_vp.save_results(model_vp.time_list, model_vp.eps)

	saver_eps_e = TensorSaver(output_folder, "eps_e")
	saver_eps_e.save_results(model_e.time_list, model_e.eps)

	saver_eps_tot = TensorSaver(output_folder, "eps_tot")
	saver_eps_tot.save_results(model_e.time_list, eps_tot)

	saver_alpha = TensorSaver(output_folder, "alpha")
	alphas = np.zeros(model_e.eps.shape)
	alphas[:,0,0] = model_vp.alphas
	saver_alpha.save_results(model_vp.time_list, alphas)

	saver_alpha_q = TensorSaver(output_folder, "alpha_q")
	alpha_qs = np.zeros(model_e.eps.shape)
	alpha_qs[:,0,0] = model_vp.alpha_qs
	saver_alpha_q.save_results(model_vp.time_list, alpha_qs)

	saver_Fvp = TensorSaver(output_folder, "Fvp")
	Fvp_list = np.zeros(model_vp.eps.shape)
	Fvp_list[:,0,0] = model_vp.Fvp_list
	saver_Fvp.save_results(model_vp.time_list, Fvp_list)


	# # Plot results
	# fig, (ax1) = plt.subplots(1, 1, figsize=(5, 4))
	# fig.subplots_adjust(top=0.935, bottom=0.125, left=0.135, right=0.985, hspace=0.2, wspace=0.2)

	# print(eps_tot[10])
	# ax1.plot(model_vp.time_list[1:]/hour, 100*eps_tot[1:,2,2], ".-", color="0.15", label=r"$\varepsilon_{axial}$", ms=8, mfc="steelblue")
	# ax1.plot(model_vp.time_list[1:]/hour, 100*eps_tot[1:,0,0], ".-", color="0.15", label=r"$\varepsilon_{radial}$", ms=8, mfc="lightcoral")
	# ax1.grid(True)

	# plt.show()


if __name__ == '__main__':
	main()
