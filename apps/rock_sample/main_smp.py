import numpy as np
import pandas as pd
import json
import os
import stat
import sys
sys.path.append(os.path.join("..", "..", "libs"))
from RockSampleSolutions import *
import time

def read_json(file_name):
	with open(file_name, "r") as j_file:
		data = json.load(j_file)
	return data

def save_json(data, file_name):
	# os.chmod(file_name, stat.S_IRWXU)
	with open(file_name, "w") as f:
	    json.dump(data, f, indent=4)

def write_stresses():
	# Read settings
	settings = read_json("settings.json")

	# Define time levels
	n_steps = 5
	time = np.linspace(0.0, 2716*hour, n_steps)
	settings["Time"]["timeList"] = list(time)

	# Define stress tensors
	settings["sigma_zz"] = list(np.repeat(18.7*MPa, n_steps))
	settings["sigma_xx"] = list(np.repeat(0.0, n_steps))
	settings["sigma_yy"] = list(np.repeat(0.0, n_steps))
	settings["sigma_xy"] = list(np.repeat(0.0, n_steps))
	settings["sigma_yz"] = list(np.repeat(0.0, n_steps))
	settings["sigma_xz"] = list(np.repeat(0.0, n_steps))

	# # Dump to file
	# save_json(settings, "settings.json")

def main():

	# Write stresses, if necessary
	# write_stresses()

	# Read settings
	input_model = read_json("input_model.json")
	input_bc = read_json("input_bc_smp.json")

	# Deine output folder
	output_folder = os.path.join(input_model["Paths"]["Output"], "smp")

	# Initialize models
	model_e = Elastic(input_model, input_bc)
	model_cr = DislocationCreep(input_model, input_bc)
	# model_ve = ViscoElastic(settings)
	# model_d = Damage(settings)
	# model_vp = ViscoPlastic_Desai(settings)

	# Compute strains
	model_e.compute_strains()
	model_cr.compute_strains()
	# model_ve.compute_strains()
	# model_d.compute_strains()
	# model_vp.compute_strains()

	# Compute total strains
	eps_tot = model_e.eps.copy()
	eps_tot += model_cr.eps.copy()
	# eps_tot += model_ve.eps.copy()
	# eps_tot += model_d.eps.copy()
	# eps_tot += model_vp.eps.copy()

	# Save results
	saver_eps_e = TensorSaver(output_folder, "eps_e")
	saver_eps_e.save_results(model_e.time_list, model_e.eps)

	# saver_eps_ve = TensorSaver(output_folder, "eps_ve")
	# saver_eps_ve.save_results(model_ve.time_list, model_ve.eps)

	saver_eps_cr = TensorSaver(output_folder, "eps_cr")
	saver_eps_cr.save_results(model_cr.time_list, model_cr.eps)

	# saver_eps_d = TensorSaver(output_folder, "eps_d")
	# saver_eps_d.save_results(model_d.time_list, model_d.eps)

	# saver_eps_vp = TensorSaver(output_folder, "eps_vp")
	# saver_eps_vp.save_results(model_vp.time_list, model_vp.eps)

	saver_eps_tot = TensorSaver(output_folder, "eps_tot")
	saver_eps_tot.save_results(model_e.time_list, eps_tot)

	# Save stresses
	stresses = [voigt2tensor(model_e.sigmas[i]) for i in range(len(model_e.time_list))]
	saver_stresses = TensorSaver(output_folder, "stresses")
	saver_stresses.save_results(model_e.time_list, stresses)

	'''
	# # Save alphas for viscoplasticity
	# saver_alpha = TensorSaver(output_folder, "alpha")
	# alphas = np.zeros(model_e.eps.shape)
	# alphas[:,0,0] = model_vp.alphas
	# saver_alpha.save_results(model_vp.time_list, alphas)

	# saver_alpha_q = TensorSaver(output_folder, "alpha_q")
	# alpha_qs = np.zeros(model_e.eps.shape)
	# alpha_qs[:,0,0] = model_vp.alpha_qs
	# saver_alpha_q.save_results(model_vp.time_list, alpha_qs)

	# # Save settings
	# save_json(settings, os.path.join(output_folder, "settings.json"))
	'''


if __name__ == '__main__':
	main()
