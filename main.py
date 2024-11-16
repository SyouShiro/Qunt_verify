import numpy as np
from dwave.system import LeapHybridCQMSampler
import dimod

# Parameters
n = 10  # Number of airspaces
m = 30  # Number of time steps

# Define routes (each route is a list of airspace indices)
routes = [
    [0, 1, 2, 3, 4],  # Route 1
    [5, 6, 7, 8, 9],  # Route 2
    [1, 2, 4, 6, 8, 4],  # Route 3 (skips some airspaces)
]

# Capacity of each airspace
c = np.array([3 for _ in range(n)])  # Default capacities
c[0] = 15
c[1] = 5
c[2] = 5
c[3] = 5
c[4] = 15

# Aircraft entering at the starting airspace of each route at each time step
u0 = {route_idx: np.zeros(m) for route_idx in range(len(routes))}
u0[0][0] = 10  # Example: Route 1 has 10 aircraft entering at time t=0
u0[1][0] = 3   # Example: Route 2 has 3 aircraft entering at time t=0
u0[2][0] = 1   # Example: Route 3 has 1 aircraft entering at time t=0

# Initialize the Constrained Quadratic Model (CQM)
cqm = dimod.ConstrainedQuadraticModel()

# Define decision variables for each route
u = {route_idx: [[dimod.Integer(f'u_{route_idx}_{i}_{t}', upper_bound=c[airspace])
                  for t in range(m + 1)] for i, airspace in enumerate(route)]
     for route_idx, route in enumerate(routes)}

x = {route_idx: [[dimod.Integer(f'x_{route_idx}_{i}_{t}', upper_bound=c[airspace])
                  for t in range(m + 1)] for i, airspace in enumerate(route)]
     for route_idx, route in enumerate(routes)}

# Add constraints for each route
# Add constraints for each route
for route_idx, route in enumerate(routes):
    for t in range(m):
        for i, airspace in enumerate(route):
            if i == 0:
                # First airspace of the route
                x_prev = u0[route_idx][t]
                u_prev = 0
            else:
                # Flow from the previous airspace in the route
                x_prev = x[route_idx][i - 1][t]
                u_prev = u[route_idx][i - 1][t]

            # Dynamic constraint: number of aircraft in each airspace at the next time step
            cqm.add_constraint(
                (x[route_idx][i][t + 1] - x_prev - u[route_idx][i][t] + u_prev) == 0,
                label=f"dynamics_constraint_route{route_idx}_airspace{i}_time{t}"
            )

            # Capacity constraints
            cqm.add_constraint(
                x[route_idx][i][t] <= c[airspace],
                label=f"capacity_constraint_x_route{route_idx}_airspace{i}_time{t}"
            )
            cqm.add_constraint(
                u[route_idx][i][t] <= c[airspace],
                label=f"capacity_constraint_u_route{route_idx}_airspace{i}_time{t}"
            )

            # Stay constraints
            cqm.add_constraint(
                x[route_idx][i][t] - u[route_idx][i][t] >= 0,
                label=f"stay_constraint_route{route_idx}_airspace{i}_time{t}"
            )

            # Non-negativity constraints
            cqm.add_constraint(
                u[route_idx][i][t] >= 0,
                label=f"nonnegativity_u_route{route_idx}_airspace{i}_time{t}"
            )
            cqm.add_constraint(
                x[route_idx][i][t] >= 0,
                label=f"nonnegativity_x_route{route_idx}_airspace{i}_time{t}"
            )


# Objective function: Minimize the total number of aircraft in the system
objective = sum(x[route_idx][i][t]
                for route_idx in range(len(routes))
                for i in range(len(routes[route_idx]))
                for t in range(m + 1))
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
    objective_value = sum(int(sample.get(f'x_{route_idx}_{i}_{t}', 0))
                          for route_idx in range(len(routes))
                          for i in range(len(routes[route_idx]))
                          for t in range(m + 1))
    print(f"Optimized objective value (total number of aircraft): {objective_value}")

    # Display the solution
    for route_idx, route in enumerate(routes):
        print(f"\nRoute {route_idx + 1}: {route}")
        # print("Optimal delay schedule u_i(t):")
        # for i, airspace in enumerate(route):
        #     print(f"  Airspace {airspace}: {[int(sample.get(f'u_{route_idx}_{i}_{t}', 0)) for t in range(m + 1)]}")

        print("Number of aircraft in each airspace x_i(t):")
        for i, airspace in enumerate(route):
            print(f"  Airspace {airspace}: {[int(sample.get(f'x_{route_idx}_{i}_{t}', 0)) for t in range(m + 1)]}")
else:
    print("No feasible solution found.")
