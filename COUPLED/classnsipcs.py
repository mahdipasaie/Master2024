import fenics as fe

class IPCSSolver:
    def __init__(self, mesh, parameters_dict, pfproblem=None):
        # Initialize mesh and parameters
        self.mesh = mesh
        self.parameters_dict = parameters_dict
        self.dt = fe.Constant(parameters_dict["dt"]) # Time step
        self.rho = fe.Constant(parameters_dict["rho_solid"])
        self.rho_solid = fe.Constant(parameters_dict["rho_solid"]) 
        self.rho_liquid = fe.Constant(parameters_dict["rho_liquid"]) 
        self.viscosity_solid = fe.Constant(parameters_dict["viscosity_solid"]) 
        self.viscosity_liquid = fe.Constant(parameters_dict["viscosity_liquid"]) 
        # The parameters also should be chnaged due to dimensionality of PF problem
        self.wscale = parameters_dict["Wscale"]
        self.tscale = parameters_dict["Tauscale"]
        self.kinsolid =(self.viscosity_solid/self.rho_solid)*(self.tscale/self.wscale**2)
        self.kinliq =(self.viscosity_liquid/self.rho_liquid)*(self.tscale/self.wscale**2)
        self.mu = self.kinliq
        self.pfproblem = pfproblem
        self.func_space()
        if self.pfproblem is not None:
            self.phi_ = self.pfproblem.phi_

    def func_space(self, degree=2):
        # Create function spaces
        self.V = fe.VectorFunctionSpace(self.mesh, 'P', degree)  
        self.Q = fe.FunctionSpace(self.mesh, 'P', degree - 1) 
        self.u = fe.TrialFunction(self.V)
        self.v = fe.TestFunction(self.V)
        self.p = fe.TrialFunction(self.Q)
        self.q = fe.TestFunction(self.Q)
        self.u_ = fe.Function(self.V)  
        self.u_s = fe.Function(self.V)  
        self.u_n = fe.Function(self.V)  
        self.u_n1 = fe.Function(self.V)  
        self.p_ = fe.Function(self.Q)  
        self.phi = fe.Function(self.Q)
        self.mufunc = fe.Function(self.Q)  
        self.phi_ = fe.Function(self.Q)  

    def form(self):


        k = self.dt
        #.1
        F1 = self.rho / k * fe.dot(self.u - self.u_n, self.v) * fe.dx
        F1 += fe.inner(fe.dot(1.5 * self.u_n - 0.5 * self.u_n1, 0.5 * fe.nabla_grad(self.u + self.u_n)), self.v) * fe.dx
        F1 += 0.5 * self.mu * fe.inner(fe.grad(self.u + self.u_n), fe.grad(self.v)) * fe.dx - fe.dot(self.p_, fe.div(self.v)) * fe.dx
        self.a1 = fe.lhs(F1)
        self.L1 = fe.rhs(F1)
        self.A1 = fe.assemble(self.a1)
        self.b1 = fe.assemble(self.L1)
        #.2
        self.a2 = fe.dot(fe.grad(self.p), fe.grad(self.q)) * fe.dx
        self.L2 = -self.rho / k * fe.dot(fe.div(self.u_s), self.q) * fe.dx
        A2 = fe.assemble(self.a2)
        b2 = fe.assemble(self.L2)   
        #.3
        self.a3 = self.rho * fe.dot(self.u, self.v) * fe.dx
        self.L3 = self.rho * fe.dot(self.u_s, self.v) * fe.dx - k * fe.dot(fe.nabla_grad(self.phi), self.v) * fe.dx
        A3 = fe.assemble(self.a3)
        b3 = fe.assemble(self.L3)

    def BC(self):

        Nx = self.parameters_dict.get("Nx")
        Ny = self.parameters_dict.get("Ny")
        inflow_profile = self.parameters_dict.get("inflow_profile")# m/s
        inflow_profile = inflow_profile*(self.tscale/self.wscale)
        inflow = 'near(x[0],0)'
        outflow = f'near(x[0],{Nx})'
        bcu_inflow = fe.DirichletBC(self.V, inflow_profile, inflow)
        bcp_outflow = fe.DirichletBC(self.Q, fe.Constant(0), outflow)
        # Assign to lists
        self.bcu = [bcu_inflow]
        self.bcp = bcp_outflow
        
    def solve(self):

        self.A1 = fe.assemble(self.a1)
        [bc.apply(self.A1) for bc in self.bcu]
        b1 = fe.assemble(self.L1)
        [bc.apply(b1) for bc in self.bcu]
        fe.solve(self.A1, self.u_s.vector(), b1, 'bicgstab', 'hypre_amg')
        # Step 2: Pressure correction step
        self.A2 = fe.assemble(self.a2)
        self.bcp.apply(self.A2)
        b2 = fe.assemble(self.L2)
        self.bcp.apply(b2)
        fe.solve(self.A2, self.phi.vector(), b2, 'bicgstab', 'hypre_amg')
        self.p_.vector()[:] += self.phi.vector()[:]  # p^{n+1} = p^{n} + phi
        # Step 3: Velocity correction step
        self.A3 = fe.assemble(self.a3)
        b3 = fe.assemble(self.L3)
        fe.solve(self.A3, self.u_.vector(), b3, "bicgstab", "hypre_amg")
        # Update previous time steps
        self.u_n.assign(self.u_)

    def inco(self):

        class InitialConditions_ns(fe.UserExpression):

            def __init__(self, **kwargs):
                super().__init__(**kwargs)  

            def eval(self, values, x):
                values[0] = 0.0  # Initial x-component of velocity
                values[1] = 0.0  # Initial y-component of velocity

            def value_shape(self):
                return (2,)
        class InitialConditions_p(fe.UserExpression):

            def __init__(self, **kwargs):
                super().__init__(**kwargs)  

            def eval(self, values, x):
                values[0] = 0.0

            def value_shape(self): 
                return ()
        
                # Define the initial conditions for velocity and pressure
        
        if self.pfproblem is not None:
            pass
        else: 
            self.initial_conditions_ns = InitialConditions_ns(degree=2)
            self.u_n.interpolate(self.initial_conditions_ns)
            self.initial_conditions_p = InitialConditions_p(degree=2)
            self.u_n1.interpolate(self.initial_conditions_ns)
            self.p_.interpolate(self.initial_conditions_p)

    def smooth_step(self, phi):

        return 3 * phi**2 - 2 * phi**3 

    def viscosity(self):

        return ( 
            (self.kinsolid) * self.smooth_step(self.phi_) 
            + (self.kinliq) * (1 - self.smooth_step(self.phi_)))
    
    def initialize(self):

        
        self.form()
        self.BC()
        self.inco()



