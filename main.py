import numpy as np
from dimod import ConstrainedQuadraticModel, Integer, quicksum
from dwave.system import LeapHybridCQMSampler
import time

# Parameters
n = 500  # Number of airspaces
m = 50 # Number of time steps - 1

# Define routes (each route is a list of airspace indices)
routes = [
    [0, 1, 2, 3, 40],  # Route 1
    [5, 6, 70, 80, 90],  # Route 2
    [100, 101, 102, 103, 104, 105],  # Route 3
    [40, 3, 2, 1, 0],  # Route 4 (reverse)
]

# Capacity of each airspace
c = np.array([3 for _ in range(n)])  # Default capacities
c[0] = 15
c[4] = 15
c[5] = 15
c[10] = 15

# Aircraft entering at the starting airspace of each route at each time step
u0 = {route_idx: np.zeros(m) for route_idx in range(len(routes))}
u0[0][0] += 6  # Route 1
u0[1][0] += 4  # Route 2
u0[2][0] += 1  # Route 3
u0[3][0] += 3  # Route 4

# Create the CQM
cqm = ConstrainedQuadraticModel()

# Route-specific decision variables
route_x = {route_idx: [[Integer(f'x_r{route_idx}_a{airspace_idx}_t{t}', lower_bound=0, upper_bound=c[airspace])
                        for t in range(m + 1)]
                       for airspace_idx, airspace in enumerate(route)]
           for route_idx, route in enumerate(routes)}

route_u = {route_idx: [[Integer(f'u_r{route_idx}_a{airspace_idx}_t{t}', lower_bound=0, upper_bound=c[airspace])
                        for t in range(m + 1)]
                       for airspace_idx, airspace in enumerate(route)]
           for route_idx, route in enumerate(routes)}

# Global variables
global_x = [[Integer(f'x_g_a{i}_t{t}', lower_bound=0, upper_bound=c[i]) for t in range(m)] for i in range(n)]
global_u = [[Integer(f'u_g_a{i}_t{t}', lower_bound=0, upper_bound=c[i]) for t in range(m)] for i in range(n)]

# Initialize route-specific x at t=0
for route_idx, route in enumerate(routes):
    for airspace_idx, airspace in enumerate(route):
        if airspace_idx == 0:
            cqm.add_constraint(route_x[route_idx][airspace_idx][0] == u0[route_idx][0],
                               label=f'init_x_r{route_idx}_a{airspace_idx}')
        else:
            cqm.add_constraint(route_x[route_idx][airspace_idx][0] == 0,
                               label=f'init_x_r{route_idx}_a{airspace_idx}')

# Flow dynamics constraints
for route_idx, route in enumerate(routes):
    for t in range(m):
        for airspace_idx, airspace in enumerate(route):
            if airspace_idx == 0:
                inflow = 0
            else:
                inflow = route_x[route_idx][airspace_idx - 1][t] - route_u[route_idx][airspace_idx - 1][t]

            # outflow = route_x[route_idx][airspace_idx][t] - route_u[route_idx][airspace_idx][t]

            # Flow dynamics constraint
            # cqm.add_constraint(
            #     route_x[route_idx][airspace_idx][t + 1]
            #     - route_x[route_idx][airspace_idx][t]
            #     - inflow
            #     + outflow == 0,
            #     label=f'flow_dynamics_r{route_idx}_a{airspace_idx}_t{t}'
            # )
            cqm.add_constraint(
                route_x[route_idx][airspace_idx][t + 1]
                - inflow
                - route_u[route_idx][airspace_idx][t] == 0,
                label=f'flow_dynamics_r{route_idx}_a{airspace_idx}_t{t}'
            )

            # Outflow cannot exceed current x
            cqm.add_constraint(
                route_u[route_idx][airspace_idx][t] >= 0,
                label=f'outflow_limit_r{route_idx}_a{airspace_idx}_t{t}'
            )
            # cqm.add_constraint(
            #     route_x[route_idx][airspace_idx][t]
            #     - outflow >= 0,
            #     label=f'outflow_limit_r{route_idx}_a{airspace_idx}_t{t}'
            # )

# Global x and u computation
for t in range(m):
    for i in range(n):
        cqm.add_constraint(
            quicksum(route_x[route_idx][route.index(i)][t]
                     for route_idx, route in enumerate(routes) if i in route) - global_x[i][t] == 0,
            label=f'global_x_a{i}_t{t}'
        )

        cqm.add_constraint(
            quicksum(route_u[route_idx][route.index(i)][t]
                     for route_idx, route in enumerate(routes) if i in route) - global_u[i][t] == 0,
            label=f'global_u_a{i}_t{t}'
        )

# Capacity constraints
for t in range(m):
    for i in range(n):
        # cqm.add_constraint(c[i] - global_x[i][t] >= 0, label=f'capacity_x_a{i}_t{t}')
        cqm.add_constraint(global_x[i][t] - global_u[i][t] >= 0, label=f'capacity_u_a{i}_t{t}')

# Objective function: Minimize total aircraft in the system
cqm.set_objective(quicksum(global_x[i][t] for i in range(n) for t in range(m)))

# Solve the CQM using D-Wave's hybrid solver
sampler = LeapHybridCQMSampler()
start_time = time.time()
min_time = sampler.min_time_limit(cqm)
give_time = min_time + 100
print("the minimum time limit for current problem is:", min_time, "seconds")
print("Solving the problem...")
sampleset = sampler.sample_cqm(cqm, time_limit=give_time)

# Process the results
feasible_samples = sampleset.filter(lambda row: row.is_feasible)
end_time = time.time()
print(f"Optimization completed in {end_time - start_time:.2f} seconds")

if feasible_samples:
    best_sample = feasible_samples.first.sample
    print("Optimal solution found!")

    # Print the total aircraft in the system
    total_aircraft = sum(best_sample[f'x_g_a{i}_t{t}'] for i in range(n) for t in range(m))
    print(f"Total aircraft in the system: {total_aircraft}")

    # Print route-specific x and u
    for route_idx, route in enumerate(routes):
        print(f"\nRoute {route_idx + 1}: {route}")
        print("Route-specific aircraft x_i(t):")
        for airspace_idx, airspace in enumerate(route):
            values = [best_sample[f'x_r{route_idx}_a{airspace_idx}_t{t}'] for t in range(m)]
            int_list = [int(item) for item in values]
            print(f"  Airspace {airspace}: {int_list}")

        print("Route-specific outflows u_i(t):")
        for airspace_idx, airspace in enumerate(route):
            values = [best_sample[f'u_r{route_idx}_a{airspace_idx}_t{t}'] for t in range(m)]
            int_list = [int(item) for item in values]
            print(f"  Airspace {airspace}: {int_list}")

    # Print global x and u
    print("\nGlobal aircraft x(t):")
    for i in range(n):
        values = [best_sample[f'x_g_a{i}_t{t}'] for t in range(m)]
        int_list = [int(item) for item in values]
        print(f"  Airspace {i}: {int_list}")

    print("\nGlobal outflows u(t):")
    for i in range(n):
        values = [best_sample[f'u_g_a{i}_t{t}'] for t in range(m)]
        int_list = [int(item) for item in values]
        print(f"  Airspace {i}: {int_list}")

else:
    print("No feasible solution found.")
