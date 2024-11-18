import cvxpy as cp
import numpy as np
import time

# Parameters
n = 16  # Number of airspaces
m = 30  # Number of time steps

# Define routes (each route is a list of airspace indices)
routes = [
    [0, 1, 2, 3, 4],  # Route 1
    [5, 6, 7, 8, 9],  # Route 2
    [10, 11, 12, 13, 14, 15],  # Route 3
    [4, 3, 2, 1, 0],  # Route 4 (reverse)
]

# Capacity of each airspace
c = np.array([3 for _ in range(n)])  # Default capacities
# Custom capacities
c[0] = 15
# c[1] = 5
# c[2] = 5
# c[3] = 5
c[4] = 15
c[5] = 15
c[10] = 15

# Aircraft entering at the starting airspace of each route at each time step
u0 = {route_idx: np.zeros(m) for route_idx in range(len(routes))}
u0[0][0] += 6  # Route 1
u0[1][0] += 4  # Route 2
u0[2][0] += 1  # Route 3
u0[3][0] += 3  # Route 4

# Route-specific decision variables
route_x = {route_idx: cp.Variable((len(route), m + 1), integer=True)
           for route_idx, route in enumerate(routes)}
route_u = {route_idx: cp.Variable((len(route), m), integer=True)
           for route_idx, route in enumerate(routes)}

# Global variables
x = cp.Variable((n, m + 1), integer=True)  # Global aircraft count in each airspace
u = cp.Variable((n, m), integer=True)      # Global outflow from each airspace

# Constraints
constraints = []

# Initialize route-specific x and u at t = 0
for route_idx, route in enumerate(routes):
    for airspace_idx, airspace in enumerate(route):
        if airspace_idx == 0:
            # First airspace in the route starts with u0
            constraints.append(route_x[route_idx][airspace_idx, 0] == u0[route_idx][0])
        else:
            # Other airspaces start with zero
            constraints.append(route_x[route_idx][airspace_idx, 0] == 0)

# Flow dynamics for each route
for route_idx, route in enumerate(routes):
    for t in range(m):
        for airspace_idx, airspace in enumerate(route):
            if airspace_idx == 0:
                # Inflow is zero (already set in init)
                inflow = 0
            else:
                # Inflow is from the previous airspace in the route
                inflow = route_x[route_idx][airspace_idx - 1, t] - route_u[route_idx][airspace_idx - 1, t]

            # Outflow from the current airspace
            outflow = route_x[route_idx][airspace_idx, t] - route_u[route_idx][airspace_idx, t]

            # Update route-specific x
            constraints.append(
                route_x[route_idx][airspace_idx, t + 1] == route_x[route_idx][airspace_idx, t] + inflow - outflow
            )

            # Ensure outflow does not exceed current aircraft
            constraints.append(outflow <= route_x[route_idx][airspace_idx, t])

            # Non-negativity constraints
            constraints.append(outflow >= 0)
            constraints.append(route_x[route_idx][airspace_idx, t] >= 0)

# Global x and u variables
for t in range(m + 1):
    for i in range(n):
        # Sum over all routes to get global x
        constraints.append(
            x[i, t] == cp.sum([route_x[route_idx][route.index(i), t]
                               for route_idx, route in enumerate(routes) if i in route])
        )

        # Non-negativity constraints for global x
        constraints.append(x[i, t] >= 0)

# Compute global u
for t in range(m):
    for i in range(n):
        # Sum over all routes to get global u
        constraints.append(
            u[i, t] == cp.sum([route_u[route_idx][route.index(i), t]
                               for route_idx, route in enumerate(routes) if i in route])
        )

        # Outflow cannot exceed current aircraft
        constraints.append(u[i, t] <= x[i, t])

        # Non-negativity constraints for global u
        constraints.append(u[i, t] >= 0)

# Capacity constraints on global x
for t in range(m + 1):
    for i in range(n):
        constraints.append(x[i, t] <= c[i])

# Objective function: Minimize the total number of aircraft in the system
objective = cp.Minimize(cp.sum(x))

# Solve the optimization problem
prob = cp.Problem(objective, constraints)

print("Solving the problem...")
start_time = time.time()
prob.solve(solver=cp.GLPK_MI)
end_time = time.time()
print(f"Optimization completed in {end_time - start_time:.2f} seconds")

# Check the problem status
print("Problem status:", prob.status)

# Output the results only if the problem is solved successfully
if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
    print("Optimal total number of aircraft in the system:", prob.value)

    for route_idx, route in enumerate(routes):
        print(f"\nRoute {route_idx + 1}: {route}")
        print("Route-specific aircraft x_i(t):")
        for airspace_idx, airspace in enumerate(route):
            int_list = [int(item) for item in route_x[route_idx][airspace_idx].value]
            print(f"  Airspace {airspace}: {int_list}")

        print("Route-specific outflows u_i(t):")
        for airspace_idx, airspace in enumerate(route):
            int_list = [int(item) for item in route_u[route_idx][airspace_idx].value]
            print(f"  Airspace {airspace}: {int_list}")

    print("\nGlobal aircraft x(t):")
    for i in range(n):
        int_list = [int(item) for item in x.value[i]]
        print(f"  Airspace {i}: {int_list}")

    print("\nGlobal outflows u(t):")
    for i in range(n):
        int_list = [int(item) for item in u.value[i]]
        print(f"  Airspace {i}: {int_list}")
else:
    print("The problem is infeasible or unbounded.")
