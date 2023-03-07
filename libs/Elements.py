from fenics import *
import numpy as np
import sympy as sy
from Utils import *
import abc

class BaseElement(metaclass=abc.ABCMeta):
	@abc.abstractmethod
	def __init__(self, fem_handler):
		self.du = fem_handler.du
		self.v = fem_handler.v
		self.dx = fem_handler.dx()
		self.TS = fem_handler.TS
		self.initialize_tensors()

	def initialize_tensors(self):
		zero_tensor = Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0)
		self.stress = local_projection(zero_tensor, self.TS)
		self.eps_tot = local_projection(zero_tensor, self.TS)

	@abc.abstractmethod
	def build_A(self, **kwargs):
		pass

	@abc.abstractmethod
	def build_b(self, **kwargs):
		pass

	# @abc.abstractmethod
	def compute_stress(self, **kwargs):
		pass

	def compute_total_strain(self, u):
		self.eps_tot.assign(local_projection(epsilon(u), self.TS))


class ElasticElement(BaseElement):
	def __init__(self, fem_handler, settings, element_name="Elastic"):
		super().__init__(fem_handler)
		self.element_name = element_name
		self.A = 0
		self.b = 0

		self.__initialize_tensors()
		self.__load_props(settings)
		self.__initialize_constitutive_matrices()

	def build_A(self):
		a_form = inner(sigma(self.C0, epsilon(self.du)), epsilon(self.v))*self.dx
		self.A = assemble(a_form)

	def build_b(self):
		pass

	def compute_stress(self):
		stress_form = voigt2stress(dot(self.C0, strain2voigt(self.eps_e)))
		self.stress.assign(local_projection(stress_form, self.TS))

	def compute_elastic_strain(self, eps_ie=None):
		eps = self.eps_tot
		if eps_ie != None:
			eps -= eps_ie
		self.eps_e.assign(local_projection(eps, self.TS))

	def __initialize_tensors(self):
		zero_tensor = Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0)
		self.eps_e = local_projection(zero_tensor, self.TS)
		self.stress = local_projection(zero_tensor, self.TS)

	def __load_props(self, settings):
		self.E = Constant(settings[self.element_name]["E"])
		self.nu = Constant(settings[self.element_name]["nu"])

	def __initialize_constitutive_matrices(self):
		self.C0_sy = constitutive_matrix_sy(self.E, self.nu)
		self.C0 = as_matrix(Constant(np.array(self.C0_sy).astype(np.float64)))


class ViscoelasticElement(BaseElement):
	def __init__(self, fem_handler, settings, element_name="Viscoelastic"):
		super().__init__(fem_handler)
		self.element_name = element_name
		self.theta = settings["Time"]["theta"]
		self.A = 0
		self.b = 0

		self.__initialize_tensors()
		self.__load_props(settings)
		self.__initialize_constitutive_matrices()

	def build_A_elastic(self):
		a_form = inner(sigma(self.C0, epsilon(self.du)), epsilon(self.v))*self.dx
		self.A_elastic = assemble(a_form)

	def build_A(self):
		a_form = inner(sigma(self.C0 + (1-self.theta)*self.C5, epsilon(self.du)), epsilon(self.v))*self.dx
		self.A = assemble(a_form)

	def build_b(self, eps_ie=None, eps_ie_old=None):
		b_form = 0
		b_form += inner(sigma(self.C4, self.eps_v_old), epsilon(self.v))*self.dx
		b_form += inner(sigma(self.theta*self.C5, self.eps_tot_old), epsilon(self.v))*self.dx
		if eps_ie != None:
			eps_ie_theta = self.theta*eps_ie_old + (1 - self.theta)*eps_ie
			b_form -= inner(sigma(self.C5, eps_ie_theta), epsilon(self.v))*self.dx
		self.b = assemble(b_form)

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

	def update(self):
		self.eps_v_old.assign(self.eps_v)
		self.eps_tot_old.assign(self.eps_tot)

	def compute_constitutive_matrices(self, dt):
		C0_bar_sy = self.__multiply(dt/self.eta, self.C0_sy)
		C1_bar_sy = self.__multiply(dt/self.eta, self.C1_sy)
		I_C0_C1_sy = self.I_sy + self.__multiply(1-self.theta, C0_bar_sy+C1_bar_sy)
		I_C0_C1_inv_sy = I_C0_C1_sy.inv()

		C2_sy = I_C0_C1_inv_sy*(self.I_sy - self.__multiply(self.theta, C0_bar_sy+C1_bar_sy))
		C3_sy = I_C0_C1_inv_sy*C0_bar_sy
		C4_sy = self.C0_sy*C2_sy
		C5_sy = self.C0_sy*C3_sy

		self.C0 = as_matrix(Constant(np.array(self.C0_sy).astype(np.float64)))
		self.C1 = as_matrix(Constant(np.array(self.C1_sy).astype(np.float64)))
		self.C2 = as_matrix(Constant(np.array(C2_sy).astype(np.float64)))
		self.C3 = as_matrix(Constant(np.array(C3_sy).astype(np.float64)))
		self.C4 = as_matrix(Constant(np.array(C4_sy).astype(np.float64)))
		self.C5 = as_matrix(Constant(np.array(C5_sy).astype(np.float64)))

	def __initialize_tensors(self,):
		zero_tensor = Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0)
		self.eps_e = local_projection(zero_tensor, self.TS)
		self.eps_v = local_projection(zero_tensor, self.TS)
		self.eps_v_old = local_projection(zero_tensor, self.TS)
		self.eps_tot_old = local_projection(zero_tensor, self.TS)

	def __load_props(self, settings):
		self.E0 = Constant(settings[self.element_name]["E0"])
		self.nu0 = Constant(settings[self.element_name]["nu0"])
		self.E1 = Constant(settings[self.element_name]["E1"])
		self.nu1 = Constant(settings[self.element_name]["nu1"])
		self.eta = Constant(settings[self.element_name]["eta"])


	def __initialize_constitutive_matrices(self):
		self.I_sy = sy.Matrix(6, 6, np.identity(6).flatten())
		self.C1_sy = constitutive_matrix_sy(self.E1, self.nu1)
		self.C1 = as_matrix(Constant(np.array(self.C1_sy).astype(np.float64)))
		self.C0_sy = constitutive_matrix_sy(self.E0, self.nu0)
		self.C0 = as_matrix(Constant(np.array(self.C0_sy).astype(np.float64)))

	def __multiply(self, a, C):
		x = sy.Symbol("x")
		return (x*C).subs(x, a)



class DashpotElement(BaseElement):
	def __init__(self, fem_handler, settings, element_name="Dashpot"):
		super().__init__(fem_handler)
		self.element_name = element_name
		self.theta = settings["Time"]["theta"]
		self.__load_props(settings)
		self.__initialize_tensors()

	def build_A(self):
		pass

	def build_b(self, C):
		b_form = inner(sigma(C, self.eps_ie), epsilon(self.v))*self.dx
		self.b = assemble(b_form)

	def update(self):
		self.eps_ie_old.assign(self.eps_ie)
		self.eps_ie_rate_old.assign(self.eps_ie_rate)

	def compute_viscous_strain(self, stress, dt):
		self.__compute_viscous_strain_rate(stress)
		self.eps_ie.assign(local_projection(self.eps_ie_old + dt*(self.theta*self.eps_ie_rate_old + (1 - self.theta)*self.eps_ie_rate), self.TS))

	def __compute_viscous_strain_rate(self, stress):
		self.eps_ie_rate.assign(local_projection((1/self.eta)*stress, self.TS))

	def __load_props(self, settings):
		self.eta = Constant(settings[self.element_name]["eta"])

	def __initialize_tensors(self,):
		zero_tensor = Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0)
		self.eps_ie = local_projection(zero_tensor, self.TS)
		self.eps_ie_old = local_projection(zero_tensor, self.TS)
		self.eps_ie_rate = local_projection(zero_tensor, self.TS)
		self.eps_ie_rate_old = local_projection(zero_tensor, self.TS)



class DislocationCreep(BaseElement):
	def __init__(self, fem_handler, settings, element_name="DislocationCreep"):
		super().__init__(fem_handler)
		self.element_name = element_name
		self.theta = settings["Time"]["theta"]
		self.__load_props(settings)
		self.__initialize_tensors()

	def build_A(self):
		pass

	def build_b(self, C):
		b_form = inner(sigma(C, self.eps_ie), epsilon(self.v))*self.dx
		self.b = assemble(b_form)

	def update(self):
		self.eps_ie_old.assign(self.eps_ie)
		self.eps_ie_rate_old.assign(self.eps_ie_rate)

	def compute_viscous_strain(self, stress, dt):
		self.__compute_viscous_strain_rate(stress)
		self.eps_ie.assign(local_projection(self.eps_ie_old + dt*(self.theta*self.eps_ie_rate_old + (1 - self.theta)*self.eps_ie_rate), self.TS))

	def __compute_viscous_strain_rate(self, stress):
		s = stress - (1./3)*tr(stress)*Identity(3)
		von_Mises = sqrt((3/2.)*inner(s, s))
		self.eps_ie_rate.assign(local_projection(self.B*(von_Mises**(self.n-1))*s, self.TS))

	def __load_props(self, settings):
		self.A = Constant(settings[self.element_name]["A"])
		self.n = Constant(settings[self.element_name]["n"])
		self.T = Constant(settings[self.element_name]["T"])
		self.R = 8.32		# Universal gas constant
		self.Q = 51600  	# Creep activation energy, [J/mol]
		self.B = float(self.A)*np.exp(-self.Q/(self.R*float(self.T)))

	def __initialize_tensors(self,):
		zero_tensor = Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0)
		self.eps_ie = local_projection(zero_tensor, self.TS)
		self.eps_ie_old = local_projection(zero_tensor, self.TS)
		self.eps_ie_rate = local_projection(zero_tensor, self.TS)
		self.eps_ie_rate_old = local_projection(zero_tensor, self.TS)


class PressureSolutionCreep(BaseElement):
	def __init__(self, fem_handler, settings, element_name="PressureSolutionCreep"):
		super().__init__(fem_handler)
		self.element_name = element_name
		self.theta = settings["Time"]["theta"]
		self.__load_props(settings)
		self.__initialize_tensors()

	def build_A(self):
		pass

	def build_b(self, C):
		b_form = inner(sigma(C, self.eps_ie), epsilon(self.v))*self.dx
		self.b = assemble(b_form)

	def update(self):
		self.eps_ie_old.assign(self.eps_ie)
		self.eps_ie_rate_old.assign(self.eps_ie_rate)

	def compute_viscous_strain(self, stress, dt):
		self.__compute_viscous_strain_rate(stress)
		self.eps_ie.assign(local_projection(self.eps_ie_old + dt*(self.theta*self.eps_ie_rate_old + (1 - self.theta)*self.eps_ie_rate), self.TS))

	def __compute_viscous_strain_rate(self, stress):
		s = stress - (1./3)*tr(stress)*Identity(3)
		self.eps_ie_rate.assign(local_projection(self.B*s, self.TS))

	def __load_props(self, settings):
		self.A = Constant(settings[self.element_name]["A"])
		self.n = Constant(settings[self.element_name]["n"])
		self.d = Constant(settings[self.element_name]["d"])
		# self.T = Constant(settings[self.element_name]["T"])
		# self.R = 8.32		# Universal gas constant
		# self.Q = 51600  	# Creep activation energy, [J/mol]
		# self.B = float(self.A)*np.exp(-self.Q/(self.R*float(self.T)))
		self.B = float(self.A)/(self.d**self.n)

	def __initialize_tensors(self,):
		zero_tensor = Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0)
		self.eps_ie = local_projection(zero_tensor, self.TS)
		self.eps_ie_old = local_projection(zero_tensor, self.TS)
		self.eps_ie_rate = local_projection(zero_tensor, self.TS)
		self.eps_ie_rate_old = local_projection(zero_tensor, self.TS)
		