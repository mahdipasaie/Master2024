import fenics as fe

class ClassPF:

    def __init__(self, params, mesh):

        self.mesh = mesh
        self.params = params
        self.dt = params["dt"]
        self.w0 = params['w0']
        self.ep4 = params['ep4']
        self.X = fe.SpatialCoordinate(self.mesh)
        self.Y = self.X[1]
        reltol = params['reltol']
        abstol = params['abstol']
        linearsolverpf = params['linearsolverpf']
        nonlinearsolverpf = params['nonlinearsolverpf']
        preconditionerpf = params['preconditionerpf']
        maximumiterationspf = params['maximumiterationspf']
        self.solver_parameters = {'nonlinear_solver': nonlinearsolverpf,
            'snes_solver': {'linear_solver': linearsolverpf,
            'report': False,"preconditioner": preconditionerpf,
            'error_on_nonconvergence': False,'absolute_tolerance': abstol,
            'relative_tolerance': reltol,'maximum_iterations': maximumiterationspf,}}

    def func_space(self, degree=2):

        P1 = fe.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)  
        P2 = fe.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)  
        element = fe.MixedElement([P1, P2])
        self.fs = fe.FunctionSpace(self.mesh, element)
        self.v_phi, self.v_c = fe.TestFunctions(self.fs)
        self.sv = fe.Function(self.fs)    
        self.sv_ = fe.Function(self.fs) 
        self.phi, self.c = fe.split(self.sv)    
        self.phi_, self.c_ = fe.split(self.sv_)        
        self.spacepf, _ = self.fs.sub(0).collapse(collapsed_dofs=True)
        self.spacec, _ = self.fs.sub(1).collapse(collapsed_dofs=True)

    def depvar(self):

        self.tol= fe.sqrt(fe.DOLFIN_EPS)  
        grad_phi = fe.grad(self.phi_)
        self.mgphi = fe.inner(grad_phi, grad_phi)
        dpx = fe.Dx(self.phi_, 0)
        dpy = fe.Dx(self.phi_, 1)
        dpx = fe.variable(dpx)
        dpy = fe.variable(dpy)
        # Normalized derivatives
        nmx = -dpx / fe.sqrt(self.mgphi)
        nmy = -dpy / fe.sqrt(self.mgphi)
        norm_phi_4 = nmx**4 + nmy**4
        an = fe.conditional(
            fe.lt(fe.sqrt(self.mgphi), self.tol),
            fe.Constant(1-3*self.ep4),
            1-3*self.ep4+ 4*self.ep4*norm_phi_4)
        wn = self.w0 * an
        self.dwnx = fe.conditional(fe.lt(fe.sqrt(self.mgphi), self.tol), 0, fe.diff(wn, dpx))
        self.dwny = fe.conditional(fe.lt(fe.sqrt(self.mgphi), self.tol), 0, fe.diff(wn, dpy))

    def form(self):
        
        term0 = (self.G* self.W_scale)*(self.Y-self.V*(self.T*self.Tscale/self.Wscale))/(self.ml* self.c0/self.keq*(1-self.keq))
        term4in = self.mgphi*self.wn*self.dwnx
        term5in = self.mgphi*self.wn*self.dwny
        term4 = -fe.inner(term4in, self.v_phi.dx(0)) 
        term5 = -fe.inner(term5in, self.v_phi.dx(1))
        term3 = -(self.wn**2*fe.inner(fe.grad(self.phi),fe.grad(self.v_phi)))
        term2 = fe.inner((self.phi - self.phi**3)-self.lamda*(self.c + term0)*(1-self.phi**2)**2, self.v_phi) 
        self.taun = (self.wn/self.w0)**2 
        term1 = -fe.inner((self.taun) * (self.phi-self.phi_) / self.dt, self.v_phi) * fe.dx
        self.eq1 = term1+term2+term3+term4+term5
        # second equation
        opk, omk= 1+self.keq, 1-self.keq 
        d = self.ds*(1+self.phi)/2+self.dl*(1-self.phi)/2
        dphidt = (self.phi-self.phi_)/self.dt
        term6 = -fe.inner(((opk) / 2 - (omk) * self.phi / 2) * (self.c - self.c_) / self.dt, self.v_c) * fe.dx
        term7 = -fe.inner(d * (1 - self.phi) / 2 * fe.grad(self.c), fe.grad(self.v_c)) * fe.dx
        term9 = (1 + (omk) * self.c) * dphidt / 2 * self.v_c * fe.dx
        self.eq2 = term6 + term7  + term9 

    def defsol(self):

        L = self.eq1 + self.eq2  
        J = fe.derivative(L, self.sol) 
        problem = fe.NonlinearVariationalProblem(L, self.sv, J=J) 
        solverpf = fe.NonlinearVariationalSolver(problem)
        self.solverpf.parameters.update(self.solver_parameters)

    def solve(self):

        self.solverpf.solve()
        self.sv_.vector()[:]= self.sv.vector()  

