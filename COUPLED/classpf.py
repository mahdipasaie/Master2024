import fenics as fe
import numpy as np

class ClassPF:

    def __init__(self, mesh, params, nsproblem= None, old_sv_=None, old_sv=None):

        self.mesh = mesh
        self.params = params
        self.dt = params["dt"]
        self.w0 = params['w0']
        self.ep4 = params['ep4']
        self.X = fe.SpatialCoordinate(self.mesh)
        self.Y = self.X[1]
        self.G = params['G']
        self.V = params['V']
        self.ml = params['ml']
        self.c0 = params['c0']
        self.keq = params['keq']
        self.lamda = params['lamda']
        self.ds = params['ds']
        self.dl = params['dl']
        self.Wscale = params['Wscale']
        self.Tscale = params['Tauscale']
        self.T = params['Time']
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

        self.dy = params['dy']
        self.y_solid = params['y_solid']
        self.old_sv_ = old_sv_
        self.old_sv = old_sv
        self.nsproblem = nsproblem
        if nsproblem is not None:
            self.u_n = nsproblem.u_n

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
        self.wn = self.w0 * an
        self.dwnx = fe.conditional(fe.lt(fe.sqrt(self.mgphi), self.tol), 0, fe.diff(self.wn, dpx))
        self.dwny = fe.conditional(fe.lt(fe.sqrt(self.mgphi), self.tol), 0, fe.diff(self.wn, dpy))

    def form(self):

        opk, omk= 1+self.keq, 1-self.keq 
        if self.nsproblem is not None:
            term1ad = - fe.inner((self.tau_n) * fe.dot(self.u_n, fe.grad(self.phi)), self.v_phi) 
            grad_phi = fe.grad(self.phi)
            # Advection Term LHS goes to RHS(Negative):V·{[(1+k-(1-k)ϕ)/2]∇U-[(1+(1-k)U)/2]∇ϕ}: 
            term10_1 = fe.dot(self.u_n,(opk-omk*self.phi)/2*fe.grad(self.c)) #V·(1+k-(1-k)ϕ)/2]∇U
            term10_2 =  fe.dot(self.u_n,-(1+omk*self.c)/2*grad_phi) # - V·[(1+(1-k)U)/2]∇ϕ
            term2ad = - fe.inner(term10_1+term10_2, self.v_c)  
        else: 
            term1ad = fe.Constant(0)*self.v_phi
            term2ad = fe.Constant(0)*self.v_c
        # first equation
        term0 = (self.G* self.Wscale)*(self.Y-self.V*(self.T*self.Tscale/self.Wscale))/(self.ml* self.c0/self.keq*(1-self.keq))
        term4in = self.mgphi*self.wn*self.dwnx
        term5in = self.mgphi*self.wn*self.dwny
        term4 = -fe.inner(term4in, self.v_phi.dx(0)) 
        term5 = -fe.inner(term5in, self.v_phi.dx(1))
        term3 = -(self.wn**2*fe.inner(fe.grad(self.phi),fe.grad(self.v_phi)))
        term2 = fe.inner((self.phi - self.phi**3)-self.lamda*(self.c + term0)*(1-self.phi**2)**2, self.v_phi) 
        self.taun = (self.wn/self.w0)**2 
        term1 = -fe.inner((self.taun) * (self.phi-self.phi_) / self.dt, self.v_phi) 
        self.eq1 = term1+term2+term3+term4+term5+term1ad 
        self.eq1 = self.eq1*fe.dx 
        # second equation
        d = self.ds*(1+self.phi)/2+self.dl*(1-self.phi)/2
        dphidt = (self.phi-self.phi_)/self.dt
        term6 = -fe.inner(((opk) / 2 - (omk) * self.phi / 2) * (self.c - self.c_) / self.dt, self.v_c) 
        term7 = -fe.inner(d * (1 - self.phi) / 2 * fe.grad(self.c), fe.grad(self.v_c)) 
        term9 = (1 + (omk) * self.c) * dphidt / 2 * self.v_c 
        self.eq2 = term6+term7+term9+term2ad
        self.eq2 = self.eq2*fe.dx

    def defsol(self):

        L = self.eq1 + self.eq2  
        J = fe.derivative(L, self.sv) 
        problem = fe.NonlinearVariationalProblem(L, self.sv, J=J) 
        self.solverpf = fe.NonlinearVariationalSolver(problem)
        self.solverpf.parameters.update(self.solver_parameters)

    def solve(self):

        self.solverpf.solve()
        self.sv_.vector()[:]= self.sv.vector()  

    def inco(self):

        class InitialConditions(fe.UserExpression):
            def __init__(self, dy, y_solid, **kwargs):
                super().__init__(**kwargs)
                self.dy = dy
                self.y_solid = y_solid
            def eval(self, values, x):
                xp = x[0]
                yp = x[1]
                perturbation_amplitude = 1*self.dy
                perturbation_wavelength = 4*self.dy  
                perturbation = perturbation_amplitude*np.sin(2 *np.pi*xp/perturbation_wavelength)
                if yp < self.y_solid - perturbation_amplitude :
                    values[0] = 1
                    values[1] = -1
                elif self.y_solid - perturbation_amplitude  <= yp <= self.y_solid + perturbation_amplitude: 
                    values[0] = perturbation 
                    values[1] = -1
                else:  # liquid
                    values[0] = -1
                    values[1] = -1
            def value_shape(self):
                return (2,)

        if self.old_sv_ is not None:

            fe.LagrangeInterpolator.interpolate(self.sv_, self.old_sv_)
            fe.LagrangeInterpolator.interpolate(self.sv, self.old_sv)

        else:
            self.sv_.interpolate(InitialConditions(self.dy, self.y_solid, degree=2))
            self.sv.interpolate(InitialConditions(self.dy, self.y_solid, degree=2))

    def initilize(self):

        self.func_space()
        self.inco()
        self.depvar()
        self.form()
        self.defsol()




        