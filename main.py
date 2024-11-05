import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel

# Define parameters
n = 3  # Number of airspaces
m = 5  # Number of time steps
c = np.array([15] + [1 for _ in range(1, n)])  # Capacities for each airspace
u0 = np.array([1] + [0 for _ in range(1, m)])  # Initial aircraft entering at airspace 0

# Helper function to create the QUBO problem
def create_qubo_problem(n, m, c, u0):
    # Initialize the QUBO dictionary
    Q = {}

    # Iterate through each time step and airspace to create constraints
    for t in range(m):
        for i in range(n):
            # Define variable indices for u and x
            u_index = f'u_{i}_{t}'
            x_index = f'x_{i}_{t}'

            # Add capacity constraints as quadratic penalties
            if i == 0:
                # Initial condition constraint for x_0
                Q[(x_index, x_index)] = 1 - u0[t] ** 2  # Numerical coefficient, not a tuple
            else:
                # Add quadratic terms and cross terms for capacity and coupling
                Q[(u_index, u_index)] = Q.get((u_index, u_index), 0) + c[i] ** 2  # Real number
                Q[(x_index, x_index)] = Q.get((x_index, x_index), 0) + c[i] ** 2  # Real number
                Q[(x_index, u_index)] = Q.get((x_index, u_index), 0) - 2 * c[i]   # Real number

    return Q

# Create the QUBO problem
qubo = create_qubo_problem(n, m, c, u0)

# Use D-Wave's sampler
sampler = EmbeddingComposite(DWaveSampler())

# Solve the problem using the quantum annealer
response = sampler.sample_qubo(qubo, num_reads=100)

# Extract the best solution
best_solution = response.first.sample
energy = response.first.energy

# Initialize arrays to store u_i(t) and x_i(t)
u = np.zeros((n, m + 1))
x = np.zeros((n, m + 1))

# Populate the arrays based on the solution
for i in range(n):
    for t in range(m + 1):
        u_var = f'u_{i}_{t}'
        x_var = f'x_{i}_{t}'
        if u_var in best_solution:
            u[i, t] = best_solution[u_var]
        if x_var in best_solution:
            x[i, t] = best_solution[x_var]

# Print the results
print("Optimal delay schedule u_i(t):")
for i in range(n):
    print(f"Airspace {i}: {u[i]}")

print("\nNumber of aircraft in each airspace x_i(t):")
for i in range(n):
    print(f"Airspace {i}: {x[i]}")

print("\nEnergy of the solution:", energy)
