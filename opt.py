import cvxpy as cp
import numpy as np

# Parameters
n = 16  # Number of airspace
m = 30  # Number of time steps

# Define routes (each route is a list of airspace indices)
routes = [
    [0, 1, 2, 3, 4],  # Route 1
    [5, 6, 7, 8, 9],  # Route 2
    [10, 11, 12, 13, 14, 15],  # Route 3 (skips some airspace)
]

# Capacity of each airspace
c = np.array([3 for _ in range(n)])  # Default capacities
# Route 1
c[0] = 15
c[1] = 5
c[2] = 5
c[3] = 5
c[4] = 15
# Route 2
c[5] = 15
# Route 3
c[10] = 15

# Aircraft entering at the starting airspace of each route at each time step
u0 = {route_idx: np.zeros(m) for route_idx in range(len(routes))}
u0[0][0] += 10  # Example: Route 1 has 10 aircraft entering at time t=0
u0[1][0] += 4  # Example: Route 2 has 4 aircraft entering at time t=0
u0[2][0] += 1  # Example: Route 3 has 1 aircraft entering at time t=0

# Decision variables
u = {route_idx: cp.Variable((len(route), m + 1), integer=True) for route_idx, route in enumerate(routes)}
x = {route_idx: cp.Variable((len(route), m + 1), integer=True) for route_idx, route in enumerate(routes)}

# Constraints
constraints = []

# Constraints for each route
for route_idx, route in enumerate(routes):
    for t in range(m):
        for airspace_idx, airspace in enumerate(route):
            if airspace_idx == 0:
                # First airspace of the route
                x_prev = u0[route_idx][t]
                u_prev = 0
            else:
                # Flow from the previous airspace in the route
                x_prev = x[route_idx][airspace_idx - 1, t]
                u_prev = u[route_idx][airspace_idx - 1, t]

            # Flow dynamics
            constraints.append(
                x[route_idx][airspace_idx, t + 1] == x_prev + u[route_idx][airspace_idx, t] - u_prev
            )

            # Capacity constraints
            constraints.append(x[route_idx][airspace_idx, t] <= c[airspace])
            constraints.append(u[route_idx][airspace_idx, t] <= c[airspace])

            # Stay constraints
            constraints.append(x[route_idx][airspace_idx, t] >= u[route_idx][airspace_idx, t])

            # Non-negativity constraints
            constraints.append(u[route_idx][airspace_idx, t] >= 0)
            constraints.append(x[route_idx][airspace_idx, t] >= 0)

# Objective function: Minimize the total number of aircraft in the system
objective = cp.Minimize(sum(cp.sum(x[route_idx]) for route_idx in range(len(routes))))

# Solve the optimization problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GLPK_MI)

# Check the problem status
print("Problem status:", prob.status)

# Output the results only if the problem is solved successfully
if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
    print("Optimal total number of aircraft in the system:", prob.value)

    for route_idx, route in enumerate(routes):
        print(f"\nRoute {route_idx + 1}: {route}")
        # print("Optimal delay schedule u_i(t):")
        # for airspace_idx, airspace in enumerate(route):
        # int_list = []
        # for item in u[route_idx].value[airspace_idx]:
        #     int_list.append(int(item))
        #     print(f"  Airspace {airspace}: {int_list}")

        print("Number of aircraft in each airspace x_i(t):")
        for airspace_idx, airspace in enumerate(route):
            int_list = []
            for item in x[route_idx].value[airspace_idx]:
                int_list.append(int(item))
            print(f"  Airspace {airspace}: {int_list}")
else:
    print("The problem is infeasible or unbounded.")
