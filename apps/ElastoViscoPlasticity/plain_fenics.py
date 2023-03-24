from dolfin import *
import sympy as sy
import numpy as np
import json

MPa = 1e6

def read_json(file_name):
    with open(file_name, "r") as j_file:
        data = json.load(j_file)
    return data

def local_projection(tensor, V):
	dv = TrialFunction(V)
	v_ = TestFunction(V)
	a_proj = inner(dv, v_)*dx
	b_proj = inner(tensor, v_)*dx
	solver = LocalSolver(a_proj, b_proj)
	solver.factorize()
	u = Function(V)
	solver.solve_local_rhs(u)
	return u

def voigt2stress(s):
    return as_matrix([[s[0], s[3], s[4]],
	    		      [s[3], s[1], s[5]],
	    		      [s[4], s[5], s[2]]])


def main():
	# Read settings json file
	settings = read_json("settings.json")

	# Load material properties
	props = settings["Viscoplastic"]
	F_0 = props["F_0"]
	mu_1 = props["mu_1"]
	N_1 = props["N_1"]
	n = props["n"]
	a_1 = props["a_1"]
	eta_1 = props["eta_1"]
	beta_1 = props["beta_1"]
	beta = props["beta"]
	m_v = props["m_v"]
	gamma = props["gamma"]
	alpha_0 = props["alpha_0"]
	alpha = alpha_0
	alpha_q = alpha
	k_v = props["k_v"]
	sigma_t = props["sigma_t"]

	# Define the number of cells in each direction
	nx = ny = nz = 3

	# Create the unit cube mesh
	mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 1), nx, ny, nz)

	# Number of elements
	n_elems = mesh.num_cells()

	# # print(mesh.cells())
	# print(mesh.num_vertices())
	# print(n_elems)

	TS = TensorFunctionSpace(mesh, "DG", 0)#, symmetry=True)
	P0 = FunctionSpace(mesh, "DG", 0)

	# Define fields
	# zero_tensor = Expression((("x[0]*x[0]","x[0]*x[1]","x[0]*x[2]"), ("x[1]*x[0]","x[1]*x[1]","x[1]*x[2]"), ("x[2]*x[0]","x[2]*x[1]","x[2]*x[2]")), degree=0)
	# stress = local_projection(Expression((("11","12","13"), ("21","22","23"), ("31","32","33")), degree=0), TS)
	# stress_field = local_projection(Expression((("1000.0","0.0","0.0"), ("0.0","1000.0","0.0"), ("0.0","0.0","5000.0")), degree=0), TS)
	stress_field = local_projection(Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","s_z")), s_z=35*MPa, degree=0), TS)
	flow_field = local_projection(Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0), TS)
	alpha_q_field = local_projection(Expression("alpha_q", alpha_q=alpha_q, degree=0), P0)
	alpha_field = local_projection(Expression("alpha", alpha=alpha, degree=0), P0)
	F_field = local_projection(Expression("0.0", degree=0), P0)
	Q_field = local_projection(Expression("0.0", degree=0), P0)

	# Modify fields
	# print(alpha.vector()[:])

	# print(stress_field.vector()[:].shape[0]/mesh.num_cells())
	# stress_field.vector()[:] = [0 for i in range(stress_field.vector()[:].size)]
	# print(stress_field.vector()[:9])

	# Define a sympy function
	s_xx = sy.Symbol("s_xx")
	s_yy = sy.Symbol("s_yy")
	s_zz = sy.Symbol("s_zz")
	s_xy = sy.Symbol("s_xy")
	s_xz = sy.Symbol("s_xz")
	s_yz = sy.Symbol("s_yz")
	a_q = sy.Symbol("a_q")

	I1 = s_xx + s_yy + s_zz
	I2 = s_xx*s_yy + s_yy*s_zz + s_xx*s_zz - s_xy**2 - s_yz**2 - s_xz**2
	I3 = s_xx*s_yy*s_zz + 2*s_xy*s_yz*s_xz - s_zz*s_xy**2 - s_xx*s_yz**2 - s_yy*s_xz**2
	J1 = 0
	J2 = (1/3)*I1**2 - I2
	J3 = (2/27)*I1**3 - (1/3)*I1*I2 + I3
	Sr = -(J3*np.sqrt(27))/(2*J2**1.5)

	I_star = I1 + sigma_t
	Q1 = (-a_q*I_star**n + gamma*I_star**2)
	Q2 = (sy.exp(beta_1*I_star) - beta*Sr)**m_v
	Qvp = J2 - Q1*Q2

	# print(sy.diff(Qvp, s_xx))

	variables = (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz, a_q)
	I1_fun = sy.lambdify(variables, I1, "numpy")
	J2_fun = sy.lambdify(variables, J2, "numpy")
	J3_fun = sy.lambdify(variables, J3, "numpy")
	Q = sy.lambdify(variables, Qvp, "numpy")
	dFdSxx = sy.lambdify(variables, sy.diff(Qvp, s_xx), "numpy")
	dFdSyy = sy.lambdify(variables, sy.diff(Qvp, s_yy), "numpy")
	dFdSzz = sy.lambdify(variables, sy.diff(Qvp, s_zz), "numpy")
	dFdSxy = sy.lambdify(variables, sy.diff(Qvp, s_xy), "numpy")
	dFdSxz = sy.lambdify(variables, sy.diff(Qvp, s_xz), "numpy")
	dFdSyz = sy.lambdify(variables, sy.diff(Qvp, s_yz), "numpy")


	# Compute F (yield function field)
	F_array = np.zeros(n_elems)
	Q_array = np.zeros(n_elems)
	flow_array = np.zeros((n_elems, 3, 3))
	for e in range(n_elems):
		ids = [9*e+0, 9*e+4, 9*e+8, 9*e+1, 9*e+2, 9*e+5]
		input_variables = np.append(stress_field.vector()[ids]/MPa, alpha_field.vector()[e])
		F_array[e] = Q(*input_variables)

		input_variables = np.append(stress_field.vector()[ids]/MPa, alpha_q_field.vector()[e])
		Q_array[e] = Q(*input_variables)

		# dqdsxx = dFdSxx(*stress_field.vector()[ids], alpha_q_field.vector()[e])
		dqdsxx = dFdSxx(*input_variables)
		dqdsyy = dFdSyy(*input_variables)
		dqdszz = dFdSzz(*input_variables)
		dqdsxy = dFdSxy(*input_variables)
		dqdsxz = dFdSxz(*input_variables)
		dqdsyz = dFdSyz(*input_variables)
		flow_array[e] = np.array([[dqdsxx, dqdsxy, dqdsxz],
								  [dqdsxy, dqdsyy, dqdsyz],
								  [dqdsxz, dqdsyz, dqdszz]])

	print(input_variables)

	F_field.vector()[:] = F_array
	Q_field.vector()[:] = Q_array
	flow_field.vector()[:] = flow_array.flatten()
	print(flow_field.vector()[:9].reshape((3,3)))
	print("Q_field:", Q_field.vector()[0])
	print("F_field:", F_field.vector()[0])
	print("I1:", I1_fun(*input_variables))
	print("J2:", J2_fun(*input_variables))
	print("J3:", J3_fun(*input_variables))


	# input_variables = np.append(stress_field.vector()[ids], alpha_field.vector()[i])

	# # input_variables = stress_field.vector()[ids].append(alpha_field.vector()[i])
	# print(input_variables)

	# print(Q(*input_variables))
	# print(dFdSxx(*input_variables))


if __name__ == '__main__':
	main()
