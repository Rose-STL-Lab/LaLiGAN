import torch
import torch.nn as nn
import sympy as sp


def SINDyConst(x):
    return torch.ones(*x.shape[:-1], 1, device=x.device)


def SINDyPoly1(x):
    return x


def SINDyPoly2(x):
    return torch.cat([
        (x[..., i] * x[..., j]).view(*x.shape[:-1], 1)
        for i in range(x.shape[-1])
        for j in range(i, x.shape[-1])],
        dim=-1)


def SINDyPoly3(x):
    return torch.cat([
        (x[..., i] * x[..., j] * x[..., k]).view(*x.shape[:-1], 1)
        for i in range(x.shape[-1])
        for j in range(i, x.shape[-1])
        for k in range(j, x.shape[-1])],
        dim=-1)


def SINDySine(x):
    return torch.sin(x)


def SINDyExp(x):
    return torch.exp(x)


class SINDyRegression(nn.Module):
    """
    Arguments:
        latent_dim: dimension of latent space
        poly_order: highest order of polynomial terms, max=3
        include_sine: whether to include sine terms
        L_list: list of Lie algebra generators
    """

    def __init__(self, latent_dim, poly_order, include_sine, include_exp, L_list=[], **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.include_exp = include_exp
        self.constraint = (len(L_list) != 0)
        self.L_list = L_list
        self.terms = []
        self.threshold = kwargs["threshold"]

        # SINDy with constraint
        if self.constraint:
            self.Q = self.get_Q().to(kwargs["device"])
            self.beta = nn.Parameter(torch.randn((self.Q.shape[1], 1), device=kwargs["device"]))
            self.const = nn.Parameter(torch.randn((latent_dim, 1), device=kwargs["device"]))
            self.Xi = self.get_Xi()
        # SINDy without constraint: all the current experiments use this setting
        else:
            self.Xi = nn.Parameter(torch.randn(self.latent_dim, self.get_term_num(), device=kwargs["device"]))
        # Mask of \Xi
        self.mask = torch.ones_like(self.Xi, device=kwargs["device"])
        # Fuction basis
        self.terms.append(SINDyConst)
        self.terms.append(SINDyPoly1)
        if poly_order > 1:
            self.terms.append(SINDyPoly2)
        if poly_order > 2:
            self.terms.append(SINDyPoly3)
        if include_sine:
            self.terms.append(SINDySine)
        if include_exp:
            self.terms.append(SINDyExp)

    def forward(self, x):
        self.Xi = self.get_Xi() if self.constraint else self.Xi
        x = torch.cat([module(x) for module in self.terms], dim=-1)
        return x @ (self.Xi * self.mask).T

    # Calculate Q, whose column space forms the null space of C
    def get_Q(self):
        M_list = self.get_M_list()
        C_list = []
        for i in range(len(M_list)):
            C = torch.kron(self.L_list[i].inverse(), M_list[i].T)
            C = C - torch.eye(C.shape[0])
            C_list.append(C)
        C_total = torch.cat(C_list, dim=0)
        U, Sigma, V = torch.svd(C_total)
        # Calculate r (rank of null space)
        for r in range(len(Sigma)):
            if abs(Sigma[-1 - r]) > 5e-3:
                break
        # Extract Q
        Q = V[:, -r:]

        # Print constraint information
        # print(f'M_list={M_list}')
        # print(f'C_total={C_total}')
        # print(f'Q={Q}')
        # print(f'Sigma={Sigma}')

        return Q

    # Calculate symbolic map M
    def get_M_list(self):
        # Create variables z0~zn-1
        z = sp.Matrix([sp.symbols(f"z{i}") for i in range(self.latent_dim)])
        # Calculate function basis library \Theta
        Theta = self.get_Theta()
        # Calculate Jacobian matrix of \Theta
        Jacobian_Theta = Theta.jacobian(z)
        # Calculate J*L*z, e.g. M*Theta
        M_temp = [Jacobian_Theta*sp.Matrix(Li)*z for Li in self.L_list]
        # Calculate M
        p = M_temp[0].shape[0]
        M_list = [torch.zeros(p, p) for i in range(len(self.L_list))]
        for i in range(len(self.L_list)):
            for j in range(p):
                expression = M_temp[i][j].expand()
                # Calculate constant term
                M_list[i][j, 0] = float(expression.subs({zi: 0 for zi in z}))
                # Calculate other terms
                for k in range(1, p):
                    # Extract coeff, using subs(z=0) to avoid bug in coeff()
                    M_list[i][j, k] = float(expression.coeff(Theta[k]).subs({zi: 0 for zi in z}))
        return M_list

    # Calculate function basis library \Theta
    def get_Theta(self):
        # Create variables z_0~z_n-1
        _ = [sp.symbols(f"z{i}") for i in range(self.latent_dim)]
        # Poly0
        Theta = sp.Matrix([1])
        # Poly1
        for i in range(self.latent_dim):
            Theta = sp.Matrix.vstack(Theta, sp.Matrix([f"z{i}"]))
        # Poly2
        if self.poly_order > 1:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    Theta = sp.Matrix.vstack(Theta, sp.Matrix([f"z{i}*z{j}"]))
        # Poly3
        if self.poly_order > 2:
            for i in range(self.latent_dim):
                for j in range(self.latent_dim):
                    for k in range(self.latent_dim):
                        Theta = sp.Matrix.vstack(Theta, sp.Matrix([f"z{i}*z{j}*z{k}"]))
        return Theta

    # Convert bata and const to Xi matrix
    def get_Xi(self):
        Xi = (self.Q @ self.beta).view(self.latent_dim, -1)
        Xi += torch.cat([self.const, torch.zeros((Xi.shape[0], Xi.shape[1]-1), device=Xi.device)], dim=1)
        return Xi

    # Get the total number of function basis
    def get_term_num(self):
        num = self.latent_dim + 1
        if self.poly_order > 1:
            num += self.latent_dim * (self.latent_dim + 1) / 2
        if self.poly_order > 2:
            num += (self.latent_dim**3 + 3*self.latent_dim**2 + 2*self.latent_dim) / 6
        if self.include_sine:
            num += self.latent_dim
        if self.include_exp:
            num += self.latent_dim
        return int(num)

    # Update mask
    def set_threshold(self, threshold):
        self.Xi = self.get_Xi() if self.constraint else self.Xi
        self.mask.data = torch.logical_and(torch.abs(self.Xi) > threshold, self.mask).float()

    # Print equations
    def print(self):
        Xi = self.get_Xi() if self.constraint else self.Xi
        for i in range(self.latent_dim):
            pos = 0
            equation = f'dz{i} ='
            # Constant term
            if self.mask[i, pos]:
                equation += f' {Xi[i, pos]:.3f} +'
            pos += 1
            # Poly1 terms
            for j in range(self.latent_dim):
                if self.mask[i, pos]:
                    equation += f' {Xi[i, pos]:.3f}*z{j} +'
                pos += 1
            # Poly2 terms
            if self.poly_order > 1:
                for j in range(self.latent_dim):
                    for k in range(j, self.latent_dim):
                        if self.mask[i, pos]:
                            equation += f' {Xi[i, pos]:.3f}*z{j}*z{k} +'
                        pos += 1
            # Poly3 terms
            if self.poly_order > 2:
                for j in range(self.latent_dim):
                    for k in range(j, self.latent_dim):
                        for l in range(k, self.latent_dim):
                            if self.mask[i, pos]:
                                equation += f' {Xi[i, pos]:.3f}*z{j}*z{k}*z{l} +'
                            pos += 1
            # Sin terms
            if self.include_sine:
                for j in range(self.latent_dim):
                    if self.mask[i, pos]:
                        equation += f' {Xi[i, pos]:.3f}*sin(z{j}) +'
                    pos += 1
            # Exp terms
            if self.include_exp:
                for j in range(self.latent_dim):
                    if self.mask[i, pos]:
                        equation += f' {Xi[i, pos]:.3f}*exp(z{j}) +'
                    pos += 1
            print(equation)
