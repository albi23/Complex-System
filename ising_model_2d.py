import math
import random as rand
from typing import List

MCS = 230_000
K0 = 30_000


def periodic_boundary_condition_step(i: int, j: int, matrix: List[List[int]], t_star: float):
    matrix_size = len(matrix)
    left: int = matrix[matrix_size - 1][j] if i == 0 else matrix[i - 1][j]
    right: int = matrix[0][j] if i == matrix_size - 1 else matrix[i + 1][j]
    up: int = matrix[i][matrix_size - 1] if j == 0 else matrix[i][j - 1]
    down: int = matrix[i][0] if j == matrix_size - 1 else matrix[i][j + 1]

    delta_energy = 2 * (matrix[i][j]) * (left + right + up + down)
    # w = math.exp(-delta_energy / t_star) # way to calculate w
    if rand.random() <= math.exp(-delta_energy / t_star):
        matrix[i][j] = -matrix[i][j]


def generate_square_matrix(size: int) -> List[List[int]]:
    return [[1 if rand.random() > 0.5 else -1 for _ in range(size)] for _ in range(size)]


def msc_steps(matrix_mode: List[List[int]], t_star: float):
    lattice_size = len(matrix_mode)
    avm = 0  # average magnetisation
    av2m = 0  # square average magnetisation
    for k in range(MCS):
        for i in range(lattice_size):
            for j in range(lattice_size):
                periodic_boundary_condition_step(i, j, matrix_mode, t_star)
        if k > K0 and k % 100 == 0:
            m = 0  # magnetisation
            for i in range(lattice_size):
                for j in range(lattice_size):
                    m += matrix_mode[i][j]
            m = m / (lattice_size * lattice_size)
            avm = avm + abs(m)
            av2m = av2m + m * m
    avm = avm / 2000
    av2m = av2m / 2000
    print(f"avm {avm} T* = {t_star}")


def two_dim_Ising_model_simulation():
    t0 = 1.5
    t_end = 3.5
    lattice_size = 10
    matrix_mode = generate_square_matrix(lattice_size)

    while t0 < t_end:
        copy = [row[:] for row in matrix_mode]
        msc_steps(copy, t0)
        t0 += 0.1


if __name__ == '__main__':
    """
    Pawlik params:
    an exemplary configurations of spins (for T*=1, T*=2.26 and T*=10);
    - calculation of averaged values (K=230 000 MCS, K0=30 000 MCS)
        - the plot of magnetization against T* (for L=5, 10, 30,60);
        - the plot of heat capacity against T* (for L=8, 16, 35) c=1/(N*T*^2)*Variance(U)
    """
    t_star = 1.7
    L = 10
    # two_dim_Ising_model_simulation(L, t_star)
    msc_steps()

"""

PUNKT 1 avg spin in a single configuration (chyba)
MLS = 230_000

DO k = 1, MCS
    avm   // średnia magnetyzacja
    DO i = 1, L
        DO j = 1, L
            M.A  s(i,j)
        END DO
    END DO
    
    if k > 30_000 and ( k  mod 100 == 0) yhen
        m = 0 // magnetyzacja
            DO i = 1, L
                DO j = 1, L
                    m = m+ s(i,j) // magnetyzacja    ----->
                END DO
            END DO
        m = m/(L*L)
        avm = avm + abs(m)
        avm2 = avm2 + m*m  // 
    endif
    avm = avm/ 2_000    // <-- pojedyńczy punkt na wykresie m od T*
    avm2 = avm2/2_000  // to do variancji zmiennej m  --> Var(m) = avm2 - avm*avm
   
END


2.  Suscebility 
do wzoru  X = L*L/(T*)  Var(m) * 1/J




3. 
"""
