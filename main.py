import numpy as np
from dwave.system import LeapHybridCQMSampler
import dimod

# Parameters
n = 10  # Number of airspace
m = 30  # Number of time steps
c = np.array([3 for _ in range(n)])  # Capacities c_i for i = 1 to n
c[0] = 15
c[1] = 5
c[2] = 5
c[3] = 5
c[4] = 15
u0 = np.array([0 for _ in range(m)])  # Aircraft entering at airspace x_0 for t = 0 to m-1
u0[0] = 10

# Initialize the Constrained Quadratic Model (CQM)
cqm = dimod.ConstrainedQuadraticModel()

# Define decision variables for aircraft delay schedule and airspace occupancy
u = [[dimod.Integer(f'u_{i}_{t}', upper_bound=c[i]) for t in range(m + 1)] for i in range(n)]
x = [[dimod.Integer(f'x_{i}_{t}', upper_bound=c[i]) for t in range(m + 1)] for i in range(n)]

# Initial conditions: x_i(0) = 0 for all i >= 1
for i in range(n):
    cqm.add_constraint(x[i][0] == 0, label=f"initial_condition_airspace_{i}")

# Dynamics and capacity constraints
for t in range(m):
    for i in range(n):
        # Determine previous values based on index
        if i == 0:
            u_prev = 0
            x_prev = u0[t]
        else:
            x_prev = x[i - 1][t]
            u_prev = u[i - 1][t]

        # Dynamic constraint: number of aircraft in each airspace at next time step
        cqm.add_constraint((x[i][t + 1] - x_prev - u[i][t] + u_prev) == 0,
                           label=f"dynamics_constraint_{i}_{t}")

        # Capacity constraints for aircraft in the airspace
        cqm.add_constraint(x[i][t] <= c[i], label=f"capacity_constraint_x_{i}_{t}")
        cqm.add_constraint(u[i][t] <= c[i], label=f"capacity_constraint_u_{i}_{t}")

        # Stay constraint reformulated as x[i][t] - u[i][t] >= 0
        cqm.add_constraint(x[i][t] - u[i][t] >= 0, label=f"stay_constraint_{i}_{t}")

        # Non-negativity constraints reformulated
        cqm.add_constraint(u[i][t] >= 0, label=f"nonnegativity_u_{i}_{t}")
        cqm.add_constraint(x[i][t] >= 0, label=f"nonnegativity_x_{i}_{t}")

# Objective function: Minimize the total number of aircraft in the system
objective = sum(x[i][t] for i in range(n) for t in range(m + 1))
cqm.set_objective(objective)

# Solve the problem using D-Wave's hybrid CQM solver
sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm, time_limit=5)

# Check if the solution is feasible
feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
if feasible_sampleset:
    sample = feasible_sampleset.first.sample
    print("Optimal solution found.")

    # Calculate the optimized value of the objective function
    objective_value = sum(int(sample.get(f'x_{i}_{t}', 0)) for i in range(n) for t in range(m + 1))
    print(f"Optimized objective value (total number of aircraft): {objective_value}")

    # Display the solution
    print("\nOptimal delay schedule u_i(t):")
    for i in range(n):
        print(f"Airspace {i}: {[int(sample.get(f'u_{i}_{t}', 0)) for t in range(m + 1)]}")

    print("\nNumber of aircraft in each airspace x_i(t):")
    for i in range(n):
        print(f"Airspace {i}: {[int(sample.get(f'x_{i}_{t}', 0)) for t in range(m + 1)]}")
else:
    print("No feasible solution found.")
