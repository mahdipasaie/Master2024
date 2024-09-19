import fenics as fe

class IPCSSolver:
    def __init__(self, mesh, parameters_dict):
        # Initialize mesh and parameters
        self.mesh = mesh
        self.parameters_dict = parameters_dict
        self.dt = fe.Constant(parameters_dict["dt"]) # Time step
        self.mu = fe.Constant(parameters_dict["mu"]) # Viscosity
        self.rho = fe.Constant(parameters_dict["rho"]) # Density
        
    def func_space(self, degree=2):
        # Create function spaces
        self.V = fe.VectorFunctionSpace(self.mesh, 'P', degree)  # Velocity space
        self.Q = fe.FunctionSpace(self.mesh, 'P', degree - 1)  # Pressure space
        # Define trial and test functions for velocity and pressure
        self.u = fe.TrialFunction(self.V)
        self.v = fe.TestFunction(self.V)
        self.p = fe.TrialFunction(self.Q)
        self.q = fe.TestFunction(self.Q)
        # Define current, previous, and intermediate velocity functions
        self.u_ = fe.Function(self.V)  # Current velocity
        self.u_s = fe.Function(self.V)  # Tentative velocity (u*)
        self.u_n = fe.Function(self.V)  # Velocity at previous time step
        self.u_n1 = fe.Function(self.V)  # Velocity from two steps back (for Adams-Bashforth)
        # Define current pressure and pressure correction
        self.p_ = fe.Function(self.Q)  # Current pressure
        self.phi = fe.Function(self.Q)  # Pressure correction (phi)


    def form(self):

        k = self.dt
        #.1
        F1 = self.rho / k * fe.dot(self.u - self.u_n, self.v) * fe.dx
        F1 += fe.inner(fe.dot(1.5 * self.u_n - 0.5 * self.u_n1, 0.5 * fe.nabla_grad(self.u + self.u_n)), self.v) * fe.dx
        F1 += 0.5 * self.mu * fe.inner(fe.grad(self.u + self.u_n), fe.grad(self.v)) * fe.dx - fe.dot(self.p_, fe.div(self.v)) * fe.dx
        # F1 += fe.dot(self.f, self.v) * fe.dx
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

        problem = self.parameters_dict.get("problem")
        if problem == "cylinder":
            inflow = 'near(x[0], 0)'
            outflow = 'near(x[0], 2.2)'
            walls = 'near(x[1], 0) || near(x[1], 0.41)'
            cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
            inflow_profile = self.parameters_dict.get("inflow_velocity")
            bc_inflow = fe.DirichletBC(self.V, inflow_profile, inflow)
            bc_walls = fe.DirichletBC(self.V, fe.Constant((0, 0)), walls)
            bc_cylinder = fe.DirichletBC(self.V, fe.Constant((0, 0)), cylinder)
            bc_outflow = fe.DirichletBC(self.Q, fe.Constant(0), outflow)
            # Assign to lists
            self.bcu = [bc_inflow, bc_walls, bc_cylinder]
            self.bcp = bc_outflow
        
        if problem == "channel":

            # Define the boundary conditions for the step flow problem
            L1 = self.parameters_dict["L1"]  # Inlet channel length
            L2 = self.parameters_dict["L2"]  # Outlet channel length
            H = self.parameters_dict["H"]    # Outlet channel height
            S = self.parameters_dict["S"]    # Step height
            # Boundary expressions
            inflow = 'near(x[0], 0)'                      # Inlet at x = 0
            outflow = f'near(x[0], {L1 + L2})'            # Outlet at x = L1 + L2
            walls = f'near(x[1], 0) || near(x[1], {H})'   # Top and bottom walls
            step_wall = f'(x[0] < {L1}) && (near(x[1], {S}))'  # Wall of the step
            vertical_step_wall = f'near(x[0], {L1}) && (x[1] < {S})'  # Vertical wall of the step
            # Define inflow velocity from parameters_dict
            inflow_profile = self.parameters_dict.get("inflow_velocity")
            # Define the boundary conditions
            bc_inflow = fe.DirichletBC(self.V, inflow_profile, inflow)  # Inlet velocity
            bc_walls = fe.DirichletBC(self.V, fe.Constant((0, 0)), walls)  # No-slip on walls
            bc_step = fe.DirichletBC(self.V, fe.Constant((0, 0)), step_wall)  # No-slip on step
            bc_outflow = fe.DirichletBC(self.Q, fe.Constant(0), outflow)  # Zero pressure at outlet
            bc_step_vertical = fe.DirichletBC(self.V, fe.Constant((0, 0)), vertical_step_wall)  # No-slip on vertical part of step
            # Assign to boundary condition lists
            self.bcu = [bc_inflow, bc_walls, bc_step, bc_step_vertical]
            self.bcp = bc_outflow


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
        fe.solve(self.A3, self.u_.vector(), b3, "cg", "sor")
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
        
        self.initial_conditions_ns = InitialConditions_ns(degree=2)
        self.u_n.interpolate(self.initial_conditions_ns)
        self.initial_conditions_p = InitialConditions_p(degree=2)
        self.u_n1.interpolate(self.initial_conditions_ns)
        self.p_.interpolate(self.initial_conditions_p)





