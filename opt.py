import cvxpy as cp
import numpy as np

# Parameters
n = 3  # Number of airspace
m = 10  # Number of time steps
c = np.array([1 for _ in range(n)])  # Capacities c_i for i = 1 to n
c[0] = 15
u0 = np.array([0 for _ in range(m)])  # Aircraft entering at airspace x_0 for t = 0 to m-1
u0[0] = 5

# Decision variables (now integers)
u = cp.Variable((n, m + 1), integer=True)
x = cp.Variable((n, m + 1), integer=True)

# Constraints
constraints = []

# Dynamics and capacity constraints
for t in range(m):
    for i in range(n):
        if i == 0:
            # For the first airspace, aircraft enter from u0[t]
            u_prev = 0
            x_prev = u0[t]
        else:
            x_prev = x[i - 1, t]
            u_prev = u[i - 1, t]

        # For the last airspace, aircraft that are not delayed will land
        # Aircraft landing: x[i, t+1] = u[i, t] + u_prev
        constraints.append(
            x[i, t + 1] == x_prev + u[i, t] - u_prev
        )

        # Capacity constraints for total aircraft in the airspace
        constraints.append(x[i, t] <= c[i])
        constraints.append(u[i, t] <= c[i])

        # stay constraints
        constraints.append(x[i, t] >= u[i, t])

        # Non-negativity constraints
        constraints.append(u[i, t] >= 0)
        constraints.append(x[i, t] >= 0)

# Objective function: Minimize the total number of aircraft in the system
objective = cp.Minimize(cp.sum(x))

# Solve the optimization problem with an appropriate solver
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GLPK_MI)

# Check the problem status
print("Problem status:", prob.status)

# Output the results only if the problem is solved successfully
if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
    print("Optimal total number of aircraft in the system:", prob.value)
    print("\nOptimal delay schedule u_i(t):")
    for i in range(n):
        print(f"Airspace {i}: {u.value[i]}")

    print("\nNumber of aircraft in each airspace x_i(t):")
    for i in range(n):
        print(f"Airspace {i}: {x.value[i]}")
else:
    print("The problem is infeasible or unbounded.")
