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

	def compute_viscous_strain(self, eps_ie=None, eps_ie_old=None):
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






class BaseViscoelasticModel():
	@abstractmethod
	def __init__(self, nu0, nu1, E0, E1, eta, theta, du, v, dx):
		self.nu0 = nu0
		self.nu1 = nu1
		self.E0 = E0
		self.E1 = E1
		self.eta = eta
		self.theta = theta
		self.du = du
		self.v = v
		self.dx = dx

	@abstractmethod
	def compute_matrices(self, dt):
		pass

	@abstractmethod
	def stress(self, eps_v_old, eps_tot, eps_tot_old):
		pass

	def constitutive_matrix_sy(self, E, nu):
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

	def initialize_matrices(self, C0_sy, C1_sy, C2_sy, C3_sy, C4_sy, C5_sy, CT_sy):
		self.C0 = as_matrix(Constant(np.array(C0_sy).astype(np.float64)))
		self.C1 = as_matrix(Constant(np.array(C1_sy).astype(np.float64)))
		self.C2 = as_matrix(Constant(np.array(C2_sy).astype(np.float64)))
		self.C3 = as_matrix(Constant(np.array(C3_sy).astype(np.float64)))
		self.C4 = as_matrix(Constant(np.array(C4_sy).astype(np.float64)))
		self.C5 = as_matrix(Constant(np.array(C5_sy).astype(np.float64)))
		self.CT = as_matrix(Constant(np.array(CT_sy).astype(np.float64)))

	def compute_viscous_strain(self, eps_v, eps_v_old, eps_tot_theta, eps_ie_theta, TS):
		A = dot(self.C2, strain2voigt(eps_v_old))
		A += dot(self.C3, strain2voigt(eps_tot_theta - eps_ie_theta))
		eps_v.assign(local_projection(voigt2stress(A), TS))


class VoigtModel(BaseViscoelasticModel):
	def __init__(self, nu0, nu1, E0, E1, eta, theta, du, v, dx, active=True):
		super().__init__(nu0, nu1, E0, E1, eta, theta, du, v, dx)
		self.name = "voigt"
		self.active = active

	def __multiply(self, a, C):
		x = sy.Symbol("x")
		return (x*C).subs(x, a)

	def compute_matrices(self, dt):
		# Define C0 and C1
		C0_sy = self.constitutive_matrix_sy(self.E0, self.nu0)
		C1_sy = self.constitutive_matrix_sy(self.E1, self.nu1)

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

		self.initialize_matrices(C0_sy, C1_sy, C2_sy, C3_sy, C4_sy, C5_sy, CT_sy)

	def build_A(self):
		if self.active:
			a_form = inner(-sigma((1 - self.theta)*self.C5, epsilon(self.du)), epsilon(self.v))*self.dx
			self.A = assemble(a_form)
		else:
			self.A = 0

	def build_b(self, eps_tot_old, eps_ie, eps_ie_old, eps_v_old):
		if self.active:
			eps_ie_theta = self.theta*eps_ie_old + (1 - self.theta)*eps_ie
			# b_form = inner(sigma(self.C0, eps_ie), epsilon(self.v))*self.dx
			b_form = -inner(sigma(self.C5, eps_ie_theta), epsilon(self.v))*self.dx
			b_form += inner(sigma(self.theta*self.C5, eps_tot_old), epsilon(self.v))*self.dx
			b_form += inner(sigma(self.C4, eps_v_old), epsilon(self.v))*self.dx
			self.b = assemble(b_form)
		else:
			self.b = 0

