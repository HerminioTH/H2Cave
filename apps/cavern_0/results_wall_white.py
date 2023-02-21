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
	fig.patch.set_facecolor("#212121ff")
	if transparent:
		fig.patch.set_alpha(0.0)
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

def apply_special_theme(fig, axes, transparent=True):
	fig.patch.set_facecolor("#212121ff")
	if transparent:
		fig.patch.set_alpha(0.0)
	for ax in axes:
		ax.grid(True, color='#c4c4c4ff')
		ax.set_axisbelow(True)
		ax.spines['bottom'].set_color('0.3')
		ax.spines['top'].set_color('0.3')
		ax.spines['right'].set_color('0.3')
		ax.spines['left'].set_color('0.3')
		ax.tick_params(axis='x', colors='0.3', which='both')
		ax.tick_params(axis='y', colors='0.3', which='both')
		ax.yaxis.label.set_color('0.3')
		ax.xaxis.label.set_color('0.3')
		ax.set_facecolor("#e9e9e9ff")


def get_max_dims(res):
	x_max = max(res.gmsh_file.points[:,0])
	y_max = max(res.gmsh_file.points[:,1])
	z_max = max(res.gmsh_file.points[:,2])
	return x_max, y_max, z_max

def get_displacement(res, fun_point):
	# Axial strain
	res.load_points_of_interest(fun_point)
	x, y, z, t, field = res.get_results_over_all_times("Displacement")
	t = np.array(t)/hour
	print(x.shape)
	print(t.shape)
	print(field.shape)
	# field = field[:,0,:]
	u = field[:,:,0]
	v = field[:,:,1]
	w = field[:,:,2]
	print(w.shape)
	return t, x, y, z, u, v, w

def save_figure(fig, output_folder, index, dpi=100):
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	fig.savefig(os.path.join(output_folder, f"fig_{index}.png"), dpi=100)


def main():
	theta = 0.5
	model_name = "voigt"
	output_folder = os.path.join("output", "case_3")
	res_A = ResultsReader(os.path.join(output_folder, "vtk"))

	alpha = -1.
	R = 45*alpha
	x_0, z_0 = 0.0, 245*alpha
	x_1, z_1 = 0.0, 415*alpha

	def wall(x, y, z):
		if np.isclose(y, 0.0, atol=1e-12, rtol=0):
			x_wall = x_0 + np.sqrt((R**2 - (z - z_0)**2))
			if abs(z) < abs(z_0-R) and np.isclose(x, 0.0, atol=1e-12, rtol=0):
				return False
			elif abs(z_0-R) <= abs(z) and abs(z) <= abs(z_0) and np.isclose(x, x_0 + np.sqrt((R**2 - (z - z_0)**2)), atol=1e-12, rtol=0):
				return True
			elif abs(z_0) <= abs(z) and abs(z) <= abs(z_1) and np.isclose(x, abs(R), atol=1e-12, rtol=0):
				return True
			elif abs(z_1) <= abs(z) and abs(z) <= abs(z_1+R) and np.isclose(x, x_1 + np.sqrt((R**2 - (z - z_1)**2)), atol=1e-12, rtol=0):
				return True
			elif abs(z) > abs(z_1-R) and np.isclose(x, 0.0, atol=1e-12, rtol=0):
				return False
			else:
				return False
		else:
			return False

	t, x, y, z, u, v, w = get_displacement(res_A, wall)
	t_day = t/24

	index_A = np.where(z == -200)[0][0]
	index_B = np.where(z == -245)[0][0]
	index_C = np.where(z == -415)[0][0]
	index_D = np.where(z == -460)[0][0]

	beta = 30
	for time_index in [-1]:
	# for time_index in range(0, len(t_day), 3):
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
		fig.subplots_adjust(top=0.94, bottom=0.125, left=0.080, right=0.98, hspace=0.21, wspace=0.26)
		dx = beta*u[time_index,:]
		dz = beta*w[time_index,:]

		ax1.plot(x, z, ".", color="0.0", label="Initial shape")
		ax1.plot(x+dx, z+dz, ".", color="0.5", label="Current shape")

		ax1.plot([x[index_A], x[index_A]+beta*u[time_index,index_A]], [z[index_A], z[index_A]+beta*w[time_index,index_A]], ".-", color="lightcoral")
		ax1.plot([x[index_B], x[index_B]+beta*u[time_index,index_B]], [z[index_B], z[index_B]+beta*w[time_index,index_B]], ".-", color="steelblue")
		ax1.plot([x[index_C], x[index_C]+beta*u[time_index,index_C]], [z[index_C], z[index_C]+beta*w[time_index,index_C]], ".-", color="mediumorchid")
		ax1.plot([x[index_D], x[index_D]+beta*u[time_index,index_D]], [z[index_D], z[index_D]+beta*w[time_index,index_D]], ".-", color="limegreen")

		ax1.text(-24, -200, "A", size=12, fontname="serif", color="lightcoral")
		ax1.text(+24, -245, "B", size=12, fontname="serif", color="steelblue")
		ax1.text(+24, -415, "C", size=12, fontname="serif", color="mediumorchid")
		ax1.text(-24, -470, "D", size=12, fontname="serif", color="limegreen")

		ax1.set_title("Time = %.1f day(s)"%t_day[time_index], size=12, fontname="serif")
		ax1.set_xlabel("Coordinate X (m)", size=12, fontname="serif")
		ax1.set_ylabel("Coordinate Z (m)", size=12, fontname="serif")
		# ax1.axis("equal")
		# ax1.set_xlim(-50, 200)
		ax1.set_xlim(-100, 250)
		ax1.set_ylim(-477, -180)
		ax1.text(108, -475, f"*Amplification factor: {beta}x", size=10, fontname="serif", color="0.4")
		ax1.legend(loc=0, fancybox=True, shadow=True)

		def s(u, v, w, i):
			return np.sqrt(u[:,i]**2 + v[:,i]**2 + w[:,i]**2)
		
		ax2.plot(t_day, s(u, v, w, index_A), ".-", color="lightcoral", label="Point A")
		ax2.plot(t_day, s(u, v, w, index_B), ".-", color="steelblue", label="Point B")
		ax2.plot(t_day, s(u, v, w, index_C), ".-", color="mediumorchid", label="Point C")
		ax2.plot(t_day, s(u, v, w, index_D), ".-", color="limegreen", label="Point D")
		ax2.grid(True)
		ax2.set_xlim(-32/24, 552/24)
		ax2.set_ylim(-0.036, 0.43)
		ax2.plot([t_day[time_index], t_day[time_index]], [-0.001, 0.43], "-", color="0.4", linewidth=1.0)
		ax2.text(t_day[time_index]-1, -0.018, "t=%.1f days"%(t_day[time_index]), color="0.4", size=10, fontname="serif")
		ax2.set_xlabel("Time (days)", size=12, fontname="serif")
		ax2.set_ylabel("Displacement (m)", size=12, fontname="serif")
		ax2.legend(loc=2, fancybox=True, shadow=True)

		# apply_dark_theme(fig, [ax1, ax2], transparent=False)
		apply_special_theme(fig, [ax1, ax2], transparent=True)

		# save_figure(fig, os.path.join(output_folder, "figures", "white"), time_index, dpi=200)

		# plt.close()

	plt.show()

if __name__ == '__main__':
	main()