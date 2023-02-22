from fenics import *
import numpy as np
import sympy as sy
from abc import ABC, abstractmethod

sec = 1.
minute = 60*sec
hour = 60*minute
day = 24*hour
month = 30*day
kPa = 1e3
MPa = 1e6
GPa = 1e9

def strain2voigt(e):
	x = 1
	return as_vector([e[0,0], e[1,1], e[2,2], x*e[0,1], x*e[0,2], x*e[1,2]])

def voigt2stress(s):
    return as_matrix([[s[0], s[3], s[4]],
		    		  [s[3], s[1], s[5]],
		    		  [s[4], s[5], s[2]]])

def epsilon(u):
	# return 0.5*(nabla_grad(u) + nabla_grad(u).T)
	# return 0.5*(grad(u) + grad(u).T)
	return sym(grad(u))

def sigma(C, eps):
	return voigt2stress(dot(C, strain2voigt(eps)))

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




class BaseModel():
	def __init__(self, du, v, dx, TensorSpace):
		self.du = du
		self.v = v
		self.dx = dx
		self.TS = TensorSpace
		self.initialize_tensors()

	def initialize_tensors(self):
		zero_tensor = Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0)
		self.stress = local_projection(zero_tensor, self.TS)
		self.eps_tot = local_projection(zero_tensor, self.TS)
		self.eps_e = local_projection(zero_tensor, self.TS)

	def build_A(self, **kwargs):
		pass

	def build_b(self, **kwargs):
		pass

	def compute_total_strain(self, u):
		self.eps_tot.assign(local_projection(epsilon(u), self.TS))


class ViscoElasticModel(BaseModel):
	def __init__(self, props, theta, du, v, dx, TensorSpace, active=True):
		super().__init__(du, v, dx, TensorSpace)
		self.props = props
		self.theta = theta
		self.name = "viscoelastic"
		self.first_call_A = True

		self.__initialize_tensors()
		self.__load_props()
		self.compute_C0_C1()

	def build_A_elastic(self):
		a_form = inner(sigma(self.C0, epsilon(self.du)), epsilon(self.v))*self.dx
		self.A_elastic = assemble(a_form)

	def build_A(self):
		if self.props["active"]:
			if self.theta != 1.0:
				a_form = inner(-sigma((1 - self.theta)*self.C5, epsilon(self.du)), epsilon(self.v))*self.dx
				# self.A += assemble(a_form)
				self.A = self.A_elastic + assemble(a_form)

	def build_b(self, eps_ie=None, eps_ie_old=None):
		if self.props["active"]:
			b_form = 0
			b_form += inner(sigma(self.theta*self.C5, self.eps_tot_old), epsilon(self.v))*self.dx
			b_form += inner(sigma(self.C4, self.eps_v_old), epsilon(self.v))*self.dx
			if eps_ie != None:
				eps_ie_theta = self.theta*eps_ie_old + (1 - self.theta)*eps_ie
				b_form -= inner(sigma(self.C5, eps_ie_theta), epsilon(self.v))*self.dx
			self.b = assemble(b_form)
		else:
			self.b = 0

	def compute_stress(self):
		stress_form = voigt2stress(dot(self.C0, strain2voigt(self.eps_e)))
		self.stress.assign(local_projection(stress_form, self.TS))

	def compute_elastic_strain(self, eps_ie=None):
		eps = self.eps_tot - self.eps_v
		if eps_ie != None:
			eps -= eps_ie
		self.eps_e.assign(local_projection(eps, self.TS))

	def compute_viscoelastic_strain(self, eps_ie=None, eps_ie_old=None):
		eps_tot_theta = self.theta*self.eps_tot_old + (1 - self.theta)*self.eps_tot
		form_v = dot(self.C2, strain2voigt(self.eps_v_old))
		form_v += dot(self.C3, strain2voigt(eps_tot_theta))
		if eps_ie != None:
			eps_ie_theta = self.theta*eps_ie_old + (1 - self.theta)*eps_ie
			form_v -= dot(self.C3, strain2voigt(eps_ie_theta))
		self.eps_v.assign(local_projection(voigt2stress(form_v), self.TS))

	def compute_C0_C1(self):
		# Define C0 and C1
		C0_sy = self.__constitutive_matrix_sy(self.E0, self.nu0)
		C1_sy = self.__constitutive_matrix_sy(self.E1, self.nu1)
		self.C0 = as_matrix(Constant(np.array(C0_sy).astype(np.float64)))
		self.C1 = as_matrix(Constant(np.array(C1_sy).astype(np.float64)))

	def compute_matrices(self, dt):
		# Define C0 and C1
		C0_sy = self.__constitutive_matrix_sy(self.E0, self.nu0)
		C1_sy = self.__constitutive_matrix_sy(self.E1, self.nu1)

		# Auxiliary 4th order tensors
		I_sy = sy.Matrix(6, 6, np.identity(6).flatten())
		C0_bar_sy = self.__multiply(dt/self.eta, C0_sy)
		C1_bar_sy = self.__multiply(dt/self.eta, C1_sy)
		I_C0_C1_sy = I_sy + self.__multiply(1-self.theta, C0_bar_sy+C1_bar_sy)
		I_C0_C1_inv_sy = I_C0_C1_sy.inv()

		C2_sy = I_C0_C1_inv_sy*(I_sy - self.__multiply(self.theta, C0_bar_sy+C1_bar_sy))
		C3_sy = I_C0_C1_inv_sy*C0_bar_sy
		C4_sy = C0_sy*C2_sy
		C5_sy = C0_sy*C3_sy
		CT_sy = C0_sy - self.__multiply(1-self.theta, C5_sy)

		self.C2 = as_matrix(Constant(np.array(C2_sy).astype(np.float64)))
		self.C3 = as_matrix(Constant(np.array(C3_sy).astype(np.float64)))
		self.C4 = as_matrix(Constant(np.array(C4_sy).astype(np.float64)))
		self.C5 = as_matrix(Constant(np.array(C5_sy).astype(np.float64)))
		self.CT = as_matrix(Constant(np.array(CT_sy).astype(np.float64)))

	def __constitutive_matrix_sy(self, E, nu):
		lame = E*nu/((1+nu)*(1-2*nu))
		G = E/(2 +2*nu)
		x = 2
		M = sy.Matrix(6, 6, [ (2*G+lame),	lame,			lame,			0.,		0.,		0.,
							  lame,			(2*G+lame),		lame,			0.,		0.,		0.,
							  lame,			lame,			(2*G+lame),		0.,		0.,		0.,
								0.,			0.,				0.,				x*G,	0.,		0.,
								0.,			0.,				0.,				0.,		x*G,	0.,
								0.,			0.,				0.,				0.,		0.,		x*G])
		return M

	def __multiply(self, a, C):
		x = sy.Symbol("x")
		return (x*C).subs(x, a)

	def __initialize_tensors(self):
		zero_tensor = Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0)
		self.eps_v = local_projection(zero_tensor, self.TS)
		self.eps_v_old = local_projection(zero_tensor, self.TS)
		self.eps_tot_old = local_projection(zero_tensor, self.TS)

	def update(self):
		self.eps_v_old.assign(self.eps_v)
		self.eps_tot_old.assign(self.eps_tot)

	def __load_props(self):
		self.E0 = Constant(self.props["E0"])
		self.E1 = Constant(self.props["E1"])
		self.nu0 = Constant(self.props["nu0"])
		self.nu1 = Constant(self.props["nu1"])
		self.eta = Constant(self.props["eta"])



class CreepDislocation():
	def __init__(self, props, theta, C_0, du, v, dx, TensorSpace):
		self.props = props
		self.du = du
		self.v = v
		self.dx = dx
		self.TS = TensorSpace
		self.theta = theta
		self.C_0 = C_0
		self.initialize_tensors()
		self.__load_props()

	def __load_props(self):
		self.A = Constant(self.props["A"])
		self.n = Constant(self.props["n"])
		self.T = Constant(self.props["T"])
		self.R = 8.32		# Universal gas constant
		self.Q = 51600  	# Creep activation energy, [J/mol]
		self.B = float(self.A)*np.exp(-self.Q/(self.R*float(self.T)))

	def initialize_tensors(self):
		zero_tensor = Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0)
		self.eps_cr = local_projection(zero_tensor, self.TS)
		self.eps_cr_old = local_projection(zero_tensor, self.TS)
		self.eps_cr_rate = local_projection(zero_tensor, self.TS)
		self.eps_cr_rate_old = local_projection(zero_tensor, self.TS)

	def update(self):
		self.eps_cr_old.assign(self.eps_cr)
		self.eps_cr_rate_old.assign(self.eps_cr_rate)

	def build_A(self):
		pass

	def build_b(self):
		if self.props["active"]:
			b_form = inner(sigma(self.C_0, self.eps_cr), epsilon(self.v))*self.dx
			self.b = assemble(b_form)
		else:
			self.b = 0

	def compute_creep_strain(self, stress, dt):
		if self.props["active"]:
			self.__compute_creep_strain_rate(stress)
			self.eps_cr.assign(local_projection(self.eps_cr_old + dt*(self.theta*self.eps_cr_rate_old + (1 - self.theta)*self.eps_cr_rate), self.TS))

	def __compute_creep_strain_rate(self, stress):
		s = stress - (1./3)*tr(stress)*Identity(3)
		von_Mises = sqrt((3/2.)*inner(s, s))
		self.eps_cr_rate.assign(local_projection(self.B*(von_Mises**(self.n-1))*s, self.TS))



class CreepPressureSolution():
	def __init__(self, A, d, T, theta, model_e, du, v, dx):
		self.du = du
		self.v = v
		self.dx = dx
		self.d = d 			# Grain size (diameter) [m]
		self.theta = theta
		self.model_e = model_e
		self.A = A
		self.T = T  		# Temperature, [K]
		self.R = 8.32		# Universal gas constant
		self.Q = 51600  	# Creep activation energy, [J/mol]
		self.B = float(self.A)/(float(self.d)**3)
		self.B *= np.exp(-self.Q/(self.R*float(self.T)))

	def build_A(self):
		pass

	def build_b(self, eps_ie):
		b_form = inner(sigma(self.model_e.C, eps_ie), epsilon(self.v))*self.dx
		self.b = assemble(b_form)

	def compute_creep_strain_rate(self, eps_cr_rate, stress, TS):
		eps_cr_rate.assign(local_projection(self.B*stress, TS))

	def compute_creep_strain(self, eps_cr, eps_cr_old, eps_cr_rate_old, eps_cr_rate, dt, TS):
		eps_cr.assign(eps_cr_old + dt*(self.theta*eps_cr_rate_old + (1 - self.theta)*eps_cr_rate))



class FemHandler():
	def __init__(self, grid):
		self.grid = grid
		self.dx = Measure("dx", domain=self.grid.mesh, subdomain_data=self.grid.subdomains)
		self.ds = Measure("ds", domain=self.grid.mesh, subdomain_data=self.grid.boundaries)
		self.V = VectorFunctionSpace(self.grid.mesh, "CG", 1)
		self.TS = TensorFunctionSpace(self.grid.mesh, "DG", 0)
		self.du = TrialFunction(self.V)
		self.v = TestFunction(self.V)
		self.v_n = dot(self.v, FacetNormal(self.grid.mesh))

class BoundaryConditionHandler():
	def __init__(self, fem_handler, settings):
		self.fem_handler = fem_handler
		self.boundaryConditions = settings["BoundaryConditions"]

	def update_Dirichlet_BC(self, time_handler):
		self.bcs = []
		index_dict = {"u_x": 0, "u_y": 1, "u_z": 2}
		for key_0, value_0 in self.boundaryConditions.items():
			u_i = index_dict[key_0]
			for BOUNDARY_NAME, VALUES in value_0.items():
				if VALUES["type"] == "DIRICHLET":
					value = Constant(VALUES["value"][time_handler.idx])
					self.bcs.append(
									DirichletBC(
												self.fem_handler.V.sub(u_i),
												value,
												self.fem_handler.grid.boundaries,
												self.fem_handler.grid.dolfin_tags[self.fem_handler.grid.boundary_dim][BOUNDARY_NAME]
									)
					)

	def update_Neumann_BC(self, time_handler):
		L_bc = 0
		for key_0, value_0 in self.boundaryConditions.items():
			for BOUNDARY_NAME, VALUES in value_0.items():
				if VALUES["type"] == "NEUMANN":
					load = Constant(VALUES["value"][time_handler.idx])
					L_bc += load*self.fem_handler.v_n*self.fem_handler.ds(self.fem_handler.grid.dolfin_tags[self.fem_handler.grid.boundary_dim][BOUNDARY_NAME])
		self.b_bc = assemble(L_bc)

	def update_BCs(self, time_handler):
		self.update_Neumann_BC(time_handler)
		self.update_Dirichlet_BC(time_handler)







class SaltModel_2():
	def __init__(self, fem_handler, bc_handler, settings):
		self.fem_handler = fem_handler
		self.bc_handler = bc_handler
		self.settings = settings
		self.initialize_solution_vector()
		self.initialize_models()

	def initialize_solution_vector(self):
		self.u = Function(self.fem_handler.V)
		self.u_k = Function(self.fem_handler.V)
		self.u.rename("Displacement", "m")
		self.u_k.rename("Displacement", "m")

	def initialize_models(self):
		self.model_v = ViscoElasticModel(
											self.settings["Viscoelastic"],
											self.settings["Time"]["theta"],
											self.fem_handler.du,
											self.fem_handler.v,
											self.fem_handler.dx,
											self.fem_handler.TS
		)

		self.model_c = CreepDislocation(
											self.settings["DislocationCreep"],
											self.settings["Time"]["theta"],
											self.model_v.C0,
											self.fem_handler.du,
											self.fem_handler.v,
											self.fem_handler.dx,
											self.fem_handler.TS
		)

	def solve_elastic_model(self, time_handler):
		# Initialize matrix
		self.model_v.build_A_elastic()
		A = self.model_v.A_elastic

		# Apply Neumann boundary conditions
		self.bc_handler.update_Neumann_BC(time_handler)
		b = self.bc_handler.b_bc

		# Solve instantaneous elastic problem
		[bc.apply(A, b) for bc in self.bc_handler.bcs]
		solve(A, self.u.vector(), b, "cg", "ilu")


	def solve_mechanics(self):
		# Initialize rhs
		b = 0

		# Build creep rhs
		self.model_c.build_b()
		b += self.model_c.b

		# Build viscoelastic rhs
		self.model_v.build_b(self.model_c.eps_cr, self.model_c.eps_cr_old)
		b += self.model_v.b

		# Boundary condition rhs
		b += self.bc_handler.b_bc

		# Apply Dirichlet boundary conditions
		[bc.apply(self.model_v.A, b) for bc in self.bc_handler.bcs]

		# Solve linear system
		solve(self.model_v.A, self.u.vector(), b, "cg", "ilu")

	def assemble_matrix(self):
		self.model_v.build_A()

	def compute_total_strain(self):
		self.model_v.compute_total_strain(self.u)

	def compute_elastic_strain(self):
		self.model_v.compute_elastic_strain(self.model_c.eps_cr)

	def compute_stress(self):
		self.model_v.compute_stress()

	def compute_inelastic_strains(self, time_handler):
		self.model_c.compute_creep_strain(self.model_v.stress, time_handler.time_step)

	def compute_viscoelastic_strain(self):
		self.model_v.compute_viscoelastic_strain(self.model_c.eps_cr, self.model_c.eps_cr_old)

	def update_old_strains(self):
		self.model_v.update()
		self.model_c.update()

	def update_matrices(self, time_handler):
		self.model_v.compute_matrices(time_handler.time_step)

	def compute_error(self):
		error = np.linalg.norm(self.u.vector() - self.u_k.vector()) / np.linalg.norm(self.u.vector())
		self.u_k.assign(self.u)
		return error



class SaltModel():
	def __init__(self, grid, settings):
		self.grid = grid
		self.settings = settings
		self.initialize()

	def initialize(self):
		self.define_measures()
		self.define_function_spaces()
		self.define_trial_test_functions()
		self.define_solution_vector()
		self.define_outward_directions()
		self.define_viscoelastic_model()
		self.define_dislocation_creep_model()
		# self.define_pressure_solution_creep_model()
		# self.define_viscoplastic_model()

	def define_measures(self):
		self.dx = Measure("dx", domain=self.grid.mesh, subdomain_data=self.grid.subdomains)
		self.ds = Measure("ds", domain=self.grid.mesh, subdomain_data=self.grid.boundaries)

	def define_function_spaces(self):
		self.V = VectorFunctionSpace(self.grid.mesh, "CG", 1)
		self.TS = TensorFunctionSpace(self.grid.mesh, "DG", 0)

	def define_trial_test_functions(self):
		self.du = TrialFunction(self.V)
		self.v = TestFunction(self.V)

	def define_outward_directions(self):
		norm = FacetNormal(self.grid.mesh)
		self.v_n = dot(self.v, norm)

	def define_viscoelastic_model(self):
		self.model_v = ViscoElasticModel(self.settings["Viscoelastic"], self.settings["Time"]["theta"], self.du, self.v, self.dx, self.TS)

	def define_dislocation_creep_model(self):
		self.model_c = CreepDislocation(self.settings["DislocationCreep"], self.settings["Time"]["theta"], self.model_v.C0, self.du, self.v, self.dx, self.TS)

	def define_solution_vector(self):
		self.u = Function(self.V)
		self.u_k = Function(self.V)
		self.u.rename("Displacement", "m")
		self.u_k.rename("Displacement", "m")

	def update_Dirichlet_BC(self, time_handler):
		self.bcs = []
		index_dict = {"u_x": 0, "u_y": 1, "u_z": 2}
		for key_0, value_0 in self.settings["BoundaryConditions"].items():
			u_i = index_dict[key_0]
			for BOUNDARY_NAME, VALUES in value_0.items():
				if VALUES["type"] == "DIRICHLET":
					value = Constant(VALUES["value"][time_handler.idx])
					self.bcs.append(DirichletBC(self.V.sub(u_i), value, self.grid.boundaries, self.grid.dolfin_tags[self.grid.boundary_dim][BOUNDARY_NAME]))

	def update_Neumann_BC(self, time_handler):
		L_bc = 0
		for key_0, value_0 in self.settings["BoundaryConditions"].items():
			for BOUNDARY_NAME, VALUES in value_0.items():
				if VALUES["type"] == "NEUMANN":
					load = Constant(VALUES["value"][time_handler.idx])
					L_bc += load*self.v_n*self.ds(self.grid.dolfin_tags[self.grid.boundary_dim][BOUNDARY_NAME])
		self.b_bc = assemble(L_bc)

	def update_BCs(self, time_handler):
		self.update_Neumann_BC(time_handler)
		self.update_Dirichlet_BC(time_handler)

	def solve_elastic_model(self, time_handler):
		# Initialize matrix
		self.model_v.build_A_elastic()
		A = self.model_v.A_elastic

		# Apply Neumann boundary conditions
		self.update_Neumann_BC(time_handler)
		b = self.b_bc

		# Solve instantaneous elastic problem
		[bc.apply(A, b) for bc in self.bcs]
		solve(A, self.u.vector(), b, "cg", "ilu")


	def solve_mechanics(self):
		# Initialize rhs
		b = 0

		# Build creep rhs
		self.model_c.build_b()
		b += self.model_c.b

		# Build viscoelastic rhs
		self.model_v.build_b(self.model_c.eps_cr, self.model_c.eps_cr_old)
		b += self.model_v.b

		# Boundary condition rhs
		b += self.b_bc

		# Apply Dirichlet boundary conditions
		[bc.apply(self.model_v.A, b) for bc in self.bcs]

		# Solve linear system
		solve(self.model_v.A, self.u.vector(), b, "cg", "ilu")

	def assemble_matrix(self):
		# Assemble matrix
		self.model_v.build_A()

	def compute_total_strain(self):
		self.model_v.compute_total_strain(self.u)

	def compute_elastic_strain(self):
		self.model_v.compute_elastic_strain(self.model_c.eps_cr)

	def compute_stress(self):
		self.model_v.compute_stress()

	def compute_creep(self, time_handler):
		self.model_c.compute_creep_strain(self.model_v.stress, time_handler.time_step)

	def compute_viscoelastic_strain(self):
		self.model_v.compute_viscoelastic_strain(self.model_c.eps_cr, self.model_c.eps_cr_old)

	def update_old_strains(self):
		self.model_v.update()
		self.model_c.update()

	def update_matrices(self, time_handler):
		self.model_v.compute_matrices(time_handler.time_step)

	def compute_error(self):
		error = np.linalg.norm(self.u.vector() - self.u_k.vector()) / np.linalg.norm(self.u.vector())
		self.u_k.assign(self.u)
		return error

