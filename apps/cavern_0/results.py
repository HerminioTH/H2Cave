import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join("..", "..", "libs"))
from ResultsHandler import ResultsReader
# from sls_maxwell import sls_maxwell

sec = 1.
minute = 60*sec
sec = 1.
hour = 60*minute
day = 24*hour
kPa = 1e3
MPa = 1e6
GPa = 1e9

def apply_dark_theme(fig, axes, transparent=True):
	fig.patch.set_facecolor("#37474fff")
	if transparent:
		fig.patch.set_alpha(0.0)
	for ax in axes:
		ax.spines['bottom'].set_color('white')
		ax.spines['top'].set_color('white')
		ax.spines['right'].set_color('white')
		ax.spines['left'].set_color('white')
		ax.tick_params(axis='x', colors='white', which='both')
		ax.tick_params(axis='y', colors='white', which='both')
		ax.yaxis.label.set_color('white')
		ax.xaxis.label.set_color('white')
		ax.title.set_color('white')
		ax.set_facecolor("#424242ff")
		ax.grid(True, color='#5a5a5aff')

def apply_white_theme(fig, axes, transparent=True):
	# fig.patch.set_facecolor("#212121ff")
	# if transparent:
	# 	fig.patch.set_alpha(0.0)
	for ax in axes:
		ax.grid(True, color='#c4c4c4ff')
		ax.set_axisbelow(True)
		ax.spines['bottom'].set_color('black')
		ax.spines['top'].set_color('black')
		ax.spines['right'].set_color('black')
		ax.spines['left'].set_color('black')
		ax.tick_params(axis='x', colors='black', which='both')
		ax.tick_params(axis='y', colors='black', which='both')
		ax.yaxis.label.set_color('black')
		ax.xaxis.label.set_color('black')
		ax.set_facecolor("#e9e9e9ff")


def get_max_dims(res):
	x_max = max(res.gmsh_file.points[:,0])
	y_max = max(res.gmsh_file.points[:,1])
	z_max = max(res.gmsh_file.points[:,2])
	return x_max, y_max, z_max

def get_displacement(res, fun_point, i):
	# Axial strain
	res.load_points_of_interest(fun_point)
	x, y, z, t, field = res.get_results_over_all_times("Displacement")
	t = np.array(t)/hour
	field = field[:,0,:]
	w = field[:,i]
	return t, w


def main():
	theta = 0.5
	model_name = "voigt"
	output_folder = os.path.join("output", "case_2", "vtk")
	res_A = ResultsReader(output_folder)

	def wall_A(x, y, z):
		if np.allclose([x, y, z], [0, 0, -200], atol=1e-5, rtol=1e-5): return True
		else: return False

	def wall_B(x, y, z):
		if np.allclose([x, y, z], [45, 0, -245], atol=1e-5, rtol=1e-5): return True
		else: return False

	def wall_C(x, y, z):
		if np.allclose([x, y, z], [45, 0, -415], atol=1e-5, rtol=1e-5): return True
		else: return False

	def wall_D(x, y, z):
		if np.allclose([x, y, z], [0, 0, -460], atol=1e-5, rtol=1e-5): return True
		else: return False

	t_A, w_A = get_displacement(res_A, wall_A, 2)
	t_B, w_B = get_displacement(res_A, wall_B, 0)
	t_C, w_C = get_displacement(res_A, wall_C, 0)
	t_D, w_D = get_displacement(res_A, wall_D, 2)



	fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
	fig.subplots_adjust(top=0.94, bottom=0.125, left=0.135, right=0.99, hspace=0.21, wspace=0.26)

	ax1.plot(t_A, w_A, ".-", color="steelblue", label="Point A")
	ax1.plot(t_B, w_B, ".-", color="lightcoral", label="Point B")
	ax1.plot(t_C, w_C, ".-", color="gold", label="Point C")
	ax1.plot(t_D, w_D, ".-", color="forestgreen", label="Point D")
	ax1.grid(True)
	ax1.set_xlabel("Time (hours)", size=12, fontname="serif")
	ax1.set_ylabel("Displacement (m)", size=12, fontname="serif")
	ax1.legend(loc=0, fancybox=True, shadow=True)

	apply_dark_theme(fig, [ax1], transparent=False)
	# # apply_white_theme(fig, axis, transparent=False)

	plt.show()

if __name__ == '__main__':
	main()