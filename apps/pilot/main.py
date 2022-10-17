from fenics import *
import os
import sys
import numpy as np
import shutil
import json

sys.path.append(os.path.join("..", "..", "libs"))
from GridHandler import GridHandler
from ResultsHandler import TensorSaver
from Models import *

sec = 1.
minute = 60*sec
hour = 60*minute
day = 24*hour
kPa = 1e3
MPa = 1e6
GPa = 1e9

def axial_stress(t):
	if t <= 911*hour:
		return 5*MPa
	elif 911*hour < t and t <= 1637.1*hour:
		return 10*MPa
	else:
		return 15*MPa

def compute_total_strain(eps_tot, u, TS):
	eps_tot.assign(local_projection(epsilon(u), TS))

def compute_elastic_strain(eps_e, eps_tot, eps_v, eps_cr, TS):
	eps_e.assign(local_projection(eps_tot - eps_cr - eps_v, TS))

def compute_theta_tensor(tensor, tensor_old, theta, TS):
	return theta*tensor_old + (1 - theta)*tensor
	# return local_projection(theta*tensor_old + (1 - theta)*tensor, TS)

def compute_error(u, u_k):
	return np.linalg.norm(u.vector() - u_k.vector()) / np.linalg.norm(u.vector())

def main():
	# Transient settings
	t = 0
	dt0 = 10*hour
	# t_final = 200*dt0
	t_final = 2500*hour
	dt = dt0
	theta = 0.5

	# Load grid
	grid_folder = os.path.join("..", "..", "grids", "quarter_cylinder_0")
	geometry_name = "geom"
	grid = GridHandler(geometry_name, grid_folder)

	# Define output folder
	output_folder = os.path.join("output", "case_0")

	# Define function space
	V = VectorFunctionSpace(grid.mesh, "CG", 1)

	# Define Dirichlet boundary conditions
	bcs = []
	bcs.append(DirichletBC(V.sub(2), Constant(0.0), grid.boundaries, grid.dolfin_tags[grid.boundary_dim]["BOTTOM"]))
	bcs.append(DirichletBC(V.sub(1), Constant(0.0), grid.boundaries, grid.dolfin_tags[grid.boundary_dim]["SIDE_Y"]))
	bcs.append(DirichletBC(V.sub(0), Constant(0.0), grid.boundaries, grid.dolfin_tags[grid.boundary_dim]["SIDE_X"]))
	# bcs.append(DirichletBC(V.sub(0), Constant(0.0), grid.boundaries, grid.dolfin_tags[grid.boundary_dim]["OUTSIDE"]))
	# bcs.append(DirichletBC(V.sub(1), Constant(0.0), grid.boundaries, grid.dolfin_tags[grid.boundary_dim]["OUTSIDE"]))

	# Define measures
	dx = Measure("dx", domain=grid.mesh, subdomain_data=grid.subdomains)
	ds = Measure("ds", domain=grid.mesh, subdomain_data=grid.boundaries)

	# Define variational problem
	du = TrialFunction(V)
	d = du.geometric_dimension()  # space dimension
	v = TestFunction(V)

	# Define normal directions on mesh
	norm = FacetNormal(grid.mesh)
	v_n = dot(v, norm)

	# Define solution vector
	u = Function(V)
	u.rename("Displacement", "m")
	u_k = Function(V)
	u_k.rename("Displacement", "m")

	# Define initial tensors
	degree = V.ufl_element().degree()
	TS = TensorFunctionSpace(grid.mesh, "DG", 0)
	zero_tensor = Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0)

	eps_e = local_projection(zero_tensor, TS)
	eps_tot = local_projection(zero_tensor, TS)
	eps_tot_old = local_projection(zero_tensor, TS)
	eps_cr = local_projection(zero_tensor, TS)
	eps_cr_old = local_projection(zero_tensor, TS)
	eps_cr_rate = local_projection(zero_tensor, TS)
	eps_cr_rate_old = local_projection(zero_tensor, TS)
	eps_v = local_projection(zero_tensor, TS)
	eps_v_old = local_projection(zero_tensor, TS)
	stress = local_projection(zero_tensor, TS)

	# Create tensor savers
	saver_eps_e = TensorSaver("eps_e", dx)
	saver_eps_tot = TensorSaver("eps_tot", dx)
	saver_eps_v = TensorSaver("eps_ve", dx)
	saver_eps_cr = TensorSaver("eps_cr", dx)
	saver_eps_cr_rate = TensorSaver("eps_cr_rate", dx)
	saver_stress = TensorSaver("stress", dx)

	# Define elastic model
	E0 = Constant(2*GPa)
	nu0 = Constant(0.3)
	model_e = ElasticModel(E0, nu0, du, v, dx)
	model_e.build_A()

	# Define viscoelastic model (Voigt)
	E1 = Constant(2*GPa)
	nu1 = Constant(0.3)
	eta = Constant(5e14)
	model_v = VoigtModel(nu0, nu1, E0, E1, eta, theta, du, v, dx)
	model_v.compute_matrices(dt)

	# Define creep model (Power-Law)
	B = Constant(5.2e-36)
	# B = Constant(0)
	n = Constant(5.0)
	T = Constant(298)
	model_c = CreepDislocation(B, n, T, theta, model_e, du, v, dx)

	# Initialize matrix
	A = model_e.A

	# Apply Neumann boundary conditions
	L_bc = -axial_stress(t)*v_n*ds(grid.dolfin_tags[grid.boundary_dim]["TOP"])
	b_bc = assemble(L_bc)
	b = b_bc

	# Solve instantaneous elastic problem
	[bc.apply(A, b) for bc in bcs]
	solve(A, u.vector(), b, "cg", "ilu")

	# Compute total strain
	compute_total_strain(eps_tot, u, TS)

	# Update total strain
	eps_tot_old.assign(eps_tot)

	# Compute elastic strain
	compute_elastic_strain(eps_e, eps_tot, eps_v, eps_cr, TS)

	# Compute stress
	model_e.compute_stress(stress, eps_e, TS)

	# Compute time integration tensors
	eps_tot_theta = compute_theta_tensor(eps_tot, eps_tot_old, theta, TS)
	eps_cr_theta = compute_theta_tensor(eps_cr, eps_cr_old, theta, TS)

	# Compute viscous strain
	model_v.compute_viscous_strain(eps_v, eps_v_old, eps_tot_theta, eps_cr_theta, TS)

	# Record tensors
	saver_eps_e.record_average(eps_e, t)
	saver_eps_v.record_average(eps_v, t)
	saver_eps_cr.record_average(eps_cr, t)
	saver_eps_cr_rate.record_average(eps_cr_rate, t)
	saver_eps_tot.record_average(eps_tot, t)
	saver_stress.record_average(stress, t)

	# Include viscous terms for transient simulation
	model_v.build_A()
	A += model_v.A

	while t < t_final:
		# Update time
		t += dt
		print(t)

		# Apply Neumann boundary conditions
		L_bc = -axial_stress(t)*v_n*ds(grid.dolfin_tags[grid.boundary_dim]["TOP"])
		b_bc = assemble(L_bc)
		b = b_bc

		# Iteration settings
		ite = 0
		tol = 1e-9
		error = 2*tol
		error_old = error

		# Compute creep strain
		model_c.compute_creep_strain(eps_cr, eps_cr_old, eps_cr_rate_old, eps_cr_rate, dt, TS)


		while error > tol:

			b = 0

			# # Build creep rhs
			model_c.build_b(eps_cr)
			b += model_c.b

			# Build viscoelastic rhs
			model_v.build_b(eps_tot_old, eps_cr, eps_cr_old, eps_v_old)
			b += model_v.b

			# Boundary condition rhs
			b += b_bc

			# Apply Dirichlet boundary conditions
			[bc.apply(A, b) for bc in bcs]

			# Solve linear system
			solve(A, u.vector(), b)
			# solve(A, u.vector(), b, "cg", "ilu")

			# Compute error
			error = compute_error(u, u_k)
			print(ite, error)

			# Update error
			error_old = error

			# Update solution
			u_k.assign(u)

			# Compute total strain
			compute_total_strain(eps_tot, u, TS)

			# Compute elastic strain
			compute_elastic_strain(eps_e, eps_tot, eps_v, eps_cr, TS)

			# Compute stress
			model_e.compute_stress(stress, eps_e, TS)

			# Compute creep strain rate
			model_c.compute_creep_strain_rate(eps_cr_rate, stress, TS)

			# Compute creep strain
			model_c.compute_creep_strain(eps_cr, eps_cr_old, eps_cr_rate_old, eps_cr_rate, dt, TS)

			# Increase iteration
			ite += 1

		# Compute time integration tensors
		eps_tot_theta = compute_theta_tensor(eps_tot, eps_tot_old, theta, TS)
		eps_cr_theta = compute_theta_tensor(eps_cr, eps_cr_old, theta, TS)

		# Compute viscous strain
		model_v.compute_viscous_strain(eps_v, eps_v_old, eps_tot_theta, eps_cr_theta, TS)

		# Update old variables
		eps_cr_rate_old.assign(eps_cr_rate)
		eps_cr_old.assign(eps_cr)
		eps_v_old.assign(eps_v)
		eps_tot_old.assign(eps_tot)

		# Record tensors
		saver_eps_e.record_average(eps_e, t)
		saver_eps_v.record_average(eps_v, t)
		saver_eps_cr.record_average(eps_cr, t)
		saver_eps_cr_rate.record_average(eps_cr_rate, t)
		saver_eps_tot.record_average(eps_tot, t)
		saver_stress.record_average(stress, t)

		# Save results
		# vtk_displacement << (u, t)
		# vtk_stress << (stress, t)
		# vtk_strain_e << (eps_e, t)
		# vtk_strain_v << (eps_v, t)
		# vtk_strain_tot << (eps_tot, t)
		# vtk_strain_cr << (eps_cr, t)
		# vtk_strain_cr_rate << (eps_cr_rate, t)

	# Save tensor results
	saver_eps_e.save(os.path.join(output_folder, "numeric"))
	saver_eps_v.save(os.path.join(output_folder, "numeric"))
	saver_eps_cr.save(os.path.join(output_folder, "numeric"))
	saver_eps_cr_rate.save(os.path.join(output_folder, "numeric"))
	saver_eps_tot.save(os.path.join(output_folder, "numeric"))
	saver_stress.save(os.path.join(output_folder, "numeric"))

	# Copy .msh mesh to output_folder
	source = os.path.join(grid_folder, geometry_name+".msh")
	destination = output_folder
	shutil.copy(source, destination)
	shutil.copy(__file__, os.path.join(destination, "copy.py"))

if __name__ == "__main__":
	main()