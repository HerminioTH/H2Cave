from dolfin import *
import sympy as sym
import json
import numpy as np

MPa = 1e6

def read_json(file_name):
    with open(file_name, "r") as j_file:
        data = json.load(j_file)
    return data

def main():

    # Define the mesh and function space
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    # Define a SymPy expression
    x, y, z = sym.symbols("x[0] x[1] x[2]")
    # x, y = sym.symbols("x0 x1")
    # expr = sym.exp(-(x**1 + y**2)/2)
    expr = -(x**1 + y**2)/2
    expr = expr.diff(x) 

    # Convert the SymPy expression to a FEniCS Expression object
    f_expr = Expression(sym.printing.ccode(expr), degree=1)

    # print(sym.printing.ccode(expr))

    # Interpolate the expression onto the function space
    f = Function(V)
    f.interpolate(f_expr)

    print(f.vector()[:])

    alpha = Constant(2.0)
    print(alpha)
    print(float(alpha))

def main_2():
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
    k_v = props["k_v"]
    sigma_t = props["sigma_t"]

    # Load stresses from boundary conditions
    bcs = settings["BoundaryConditions"]
    sigmas_xx = np.array(bcs["u_x"]["OUTSIDE"]["value"])/MPa
    # sigmas_yy = np.array(bcs["u_y"]["OUTSIDE"]["value"])/MPa
    sigmas_yy = sigmas_xx
    sigmas_zz = np.array(bcs["u_z"]["TOP"]["value"])/MPa
    sigmas_xy = sigmas_xz = sigmas_yz = np.array([0.0 for i in range(len(sigmas_zz))])

    # sigmas_xx = -sigmas_xx
    # sigmas_yy = -sigmas_yy
    # sigmas_zz = -sigmas_zz

    # Compute stress invariants
    I1 = sigmas_xx + sigmas_yy + sigmas_zz
    I2 = sigmas_xx*sigmas_yy + sigmas_yy*sigmas_zz + sigmas_xx*sigmas_zz - sigmas_xy**2 - sigmas_yz**2 - sigmas_xz**2
    I3 = sigmas_xx*sigmas_yy*sigmas_zz + 2*sigmas_xy*sigmas_yz*sigmas_xz - sigmas_zz*sigmas_xy**2 - sigmas_xx*sigmas_yz**2 - sigmas_yy*sigmas_xz**2
    print("I1: ", I1[0])
    print("I2: ", I2[0])
    print("I3: ", I3[0])

    J1 = np.array([0.0 for i in range(len(sigmas_zz))])
    J2 = (1/3)*I1**2 - I2
    J3 = (2/27)*I1**3 - (1/3)*I1*I2 + I3
    print("J1: ", J1[0])
    print("J2: ", J2[0])
    print("J3: ", J3[0])

    I1_star = I1 + sigma_t
    Sr = -(J3*np.sqrt(27))/(2*J2**1.5)

    # Compute yield function
    print("n: ", n)
    print("gamma: ", gamma)
    print("beta_1: ", beta_1)
    print("beta: ", beta)
    print("m_v: ", m_v)
    print("alpha: ", alpha)
    F1 = (-alpha*I1_star**n + gamma*I1_star**2)
    F2 = (np.exp(beta_1*I1_star) - beta*Sr)**m_v
    Fvp = J2 - F1*F2
    print("Fvp: ", Fvp)

    # Sr = -(J3*np.sqrt(27))/(2*J2**1.5)
    # F1 = (-self.alpha*I1_star**self.n + self.gamma*I1_star**2)
    # F2 = (np.exp(self.beta_1*I1_star) - self.beta*Sr)**self.m
    # self.Fvp = J2 - F1*F2


def main_3():

    A = np.random.rand(3, 3)
    B = np.random.rand(3, 3)

    def double_dot_1(A, B):
        return np.tensordot(A, B.T, axes=2)

    def double_dot_2(A, B):
        n, m = A.shape
        value = 0.0
        for i in range(n):
            for j in range(m):
                value += A[i,j]*B[j,i]
        return value

    print(A)
    print(B)
    print(double_dot_1(A,B))
    print(double_dot_2(A,B))

    C = 2*np.eye(3)
    print()
    print(double_dot_1(C,C))
    print(C[0,0])
    

if __name__ == "__main__":
    # main()
    # main_2()
    main_3()