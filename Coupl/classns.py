import fenics as fe
import numpy as np

class NS:

    def __init__(self, mesh, parameters_dict, nsproblem=None ,pfproblem=None):

        self.mesh = mesh
        self.nsproblem = nsproblem
        self.pfproblem = pfproblem
        self.parameters_dict = parameters_dict
        self.dt = fe.Constant(parameters_dict["dt"]) # Time step
        self.rho = fe.Constant(parameters_dict["rho_solid"])
        self.rho_solid = fe.Constant(parameters_dict["rho_solid"]) 
        self.rho_liquid = fe.Constant(parameters_dict["rho_liquid"]) 
        self.viscosity_solid = fe.Constant(parameters_dict["viscosity_solid"]) 
        self.viscosity_liquid = fe.Constant(parameters_dict["viscosity_liquid"]) 
        self.wscale = parameters_dict["Wscale"]
        self.wscale = parameters_dict["Wscale"]
        self.tscale = parameters_dict["Tauscale"]
        self.kinsolid =(self.viscosity_solid/self.rho_solid)*(self.tscale/self.wscale**2)
        self.kinliq =(self.viscosity_liquid/self.rho_liquid)*(self.tscale/self.wscale**2)
        self.mu = self.kinliq
        self.pfproblem = pfproblem
        self.solver_parameters = {
            'nonlinear_solver': self.parameters_dict["nonlinearsolverns"],
            'snes_solver': {
                'linear_solver': self.parameters_dict["linearsolverns"],
                'report': False,
                "preconditioner": self.parameters_dict["preconditionerns"],
                'error_on_nonconvergence': False,
                'absolute_tolerance': self.parameters_dict["abs_tol_ns"],
                'relative_tolerance': self.parameters_dict["rel_tol_ns"],
                'maximum_iterations': self.parameters_dict["maximumiterationsns"],
            }
        }
        if self.nsproblem is not None:
            self.old_sv_ = self.nsproblem.sv_ 
            self.old_sv = self.nsproblem.sv
        if self.pfproblem is not None:
            self.phi_, self.c_ = self.pfproblem.sv_.split(deepcopy=True) # use self.phi_interp
        # Initialize 
        self.func_space()
        self.form()
        self.BC()
        self.InitialC()
        self.solver()

    def func_space(self, degree=2):

        P1 = fe.VectorElement("Lagrange", self.mesh.ufl_cell(), 2)  
        P2 = fe.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1) 
        EL = fe.MixedElement([P1, P2])
        self.fs = fe.FunctionSpace(self.mesh, EL)
        self.v, self.q = fe.TestFunctions(self.fs)
        self.sv = fe.Function(self.fs)  
        self.sv_ = fe.Function(self.fs)  
        self.u, self.p = fe.split(self.sv)  
        self.u_, self.p_ = fe.split(self.sv_) 
        self.space_u, _ = self.fs.sub(0).collapse(collapsed_dofs=True)
        self.space_p, _ = self.fs.sub(1).collapse(collapsed_dofs=True)
        self.phi_interp = fe.Function(self.space_p)
        if self.pfproblem is not None:
            fe.LagrangeInterpolator.interpolate(self.phi_interp, self.phi_)

    def form(self):


        beta = 1e6  # Adjust as needed
        epsilon = 0.05
        H = self.smooth_heaviside(self.phi_interp, epsilon)
        penalization = beta * H * fe.inner(self.u, self.v) * fe.dx
        Fp = fe.inner(fe.div(self.u)/self.dt, self.q)  
        Fu = (
            fe.inner((self.u - self.u_)/self.dt, self.v) 
            + fe.inner(fe.dot(self.u,fe.grad(self.u)),self.v) 
            + self.mu*fe.inner(self.sigma(self.u, self.p), self.epsilon(self.v)) 
            + fe.inner(penalization_term, self.v)           
        )
        self.F = (Fp + Fu)*fe.dx

    def epsilon(self,u):  
        return 0.5*(fe.grad(u)+fe.grad(u).T)
    
    def sigma(self,u, p):
        return 2*self.epsilon(u) - p*fe.Identity(len(u))
    
    def BC(self):

        Nx = self.parameters_dict.get("Nx")
        Ny = self.parameters_dict.get("Ny")
        max_y = self.parameters_dict.get("y_solid")
        velx = self.parameters_dict.get("velx")# m/s
        velx = velx*self.tscale/self.wscale
        # # Parabolic inflow profile: velocity is zero below max_y, parabolic above max_y
        # inflow_profile = fe.Expression(('(x[1] >= max_y) ? velx * (1 - pow((x[1] - max_y) / (Ny - max_y), 2)) : 0.0', '0.0'),
        #                            velx=velx, Ny=Ny, max_y=max_y, degree=2)

        inflow_profile = (fe.Expression((
            'velx * (tanh((x[1] - max_y) / eps) + 1.0) / 2.0 * (1.0 - pow((x[1] - max_y) / (Ny - max_y), 2))',
            '0.0'),
            velx=velx, Ny=Ny, max_y=max_y, eps=0.05, degree=2))

        inflow = 'near(x[0],0)'
        outflow = f'near(x[0],{Nx})'
        walls = f'near(x[1],0) || near(x[1],{Ny})'
        bcu_inflow = fe.DirichletBC(self.fs.sub(0), inflow_profile, inflow)
        bcp_outflow = fe.DirichletBC(self.fs.sub(1), fe.Constant(0), outflow)
        bc_walls = fe.DirichletBC(self.fs.sub(0), fe.Constant((0.0, 0.0)), walls)
        self.Bc = [bcu_inflow, bcp_outflow, bc_walls]

    def solver(self):

        J= fe.derivative(self.F, self.sv)
        self.problem= fe.NonlinearVariationalProblem(self.F, self.sv, self.Bc, J)
        self.solver = fe.NonlinearVariationalSolver(self.problem)
        self.solver.parameters.update(self.solver_parameters)

    def InitialC(self):

        class InitialConditions_ns(fe.UserExpression):

            def __init__(self, params, **kwargs):
                super().__init__(**kwargs)  

            def eval(self, values, x):
                values[0] = 0.0  
                values[1] = 0.0  
                values[2] = 0.0  

            def value_shape(self):
                return (3,)
            
        if self.nsproblem is not None:
            fe.LagrangeInterpolator.interpolate(self.sv_, self.old_sv_)
            fe.LagrangeInterpolator.interpolate(self.sv, self.old_sv)
        else:
            self.sv_.interpolate(InitialConditions_ns(self.parameters_dict, degree=2))
            self.sv.interpolate(InitialConditions_ns(self.parameters_dict, degree=2))

    def solve(self):

        self.solver.solve()
        self.sv_.vector()[:]= self.sv.vector()

    def smooth_step(self, phi):

        return 3 * phi**2 - 2 * phi**3 

    def viscosity(self):

        return ( 
            (self.kinsolid) * self.smooth_step(self.phi_interp) 
            + (self.kinliq) * (1 - self.smooth_step(self.phi_interp)))
    