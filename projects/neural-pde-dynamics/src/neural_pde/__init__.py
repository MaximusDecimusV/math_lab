from neural_pde.burgers import BurgersSpectral, cole_hopf_solution, energy, mass
from neural_pde.pinn import BurgersPINN, LaplaceTrial, laplace_exact, laplace_fd_solve

__all__ = [
    "BurgersPINN",
    "BurgersSpectral",
    "LaplaceTrial",
    "cole_hopf_solution",
    "energy",
    "laplace_exact",
    "laplace_fd_solve",
    "mass",
]
