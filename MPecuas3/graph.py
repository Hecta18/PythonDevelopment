import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the model
def model(Y, t):
    x, y = Y
    dxdt = -0.1 * y
    dydt = -0.064 * x
    return [dxdt, dydt]

# Initial conditions
Y0_1 = [10, 6]
Y0_2 = [10, 7]
Y0_3 = [10, 8]
Y0_4 = [10, 9]
Y0_5 = [10, 10]

# Time points
t = np.linspace(0, 60)

# Solve ODEs with adaptive step-size algorithms (like Runge-Kutta methods)
sol1 = odeint(model, Y0_1, t)
sol2 = odeint(model, Y0_2, t)
sol3 = odeint(model, Y0_3, t)
sol4 = odeint(model, Y0_4, t)
sol5 = odeint(model, Y0_5, t)

# Function to find where solution reaches axes (x=0 or y=0)
def truncate_at_axes(solution, time_array):
    x_vals, y_vals = solution.T
    
    # Find first index where x or y becomes <= 0
    x_zero_idx = np.where(x_vals <= 0)[0]
    y_zero_idx = np.where(y_vals <= 0)[0]
    
    # Get the minimum index (first crossing)
    zero_indices = []
    if len(x_zero_idx) > 0:
        zero_indices.append(x_zero_idx[0])
    if len(y_zero_idx) > 0:
        zero_indices.append(y_zero_idx[0])
    
    if zero_indices:
        cutoff_idx = min(zero_indices)
        return x_vals[:cutoff_idx+1], y_vals[:cutoff_idx+1], time_array[:cutoff_idx+1]
    else:
        return x_vals, y_vals, time_array

# Extracting solutions and truncating at axes
x1, y1, t1 = truncate_at_axes(sol1, t)
x2, y2, t2 = truncate_at_axes(sol2, t)
x3, y3, t3 = truncate_at_axes(sol3, t)
x4, y4, t4 = truncate_at_axes(sol4, t)
x5, y5, t5 = truncate_at_axes(sol5, t)

plt.figure(figsize=(12, 6))

# Plotting the curves
plt.subplot(1, 2, 1)
# plt.plot(t1, y1, label='y(0)=6', color='blue')
# plt.plot(t2, y2, label='y(0)=7', color='orange')
# plt.plot(t3, y3, label='y(0)=8', color='green')
# plt.plot(t4, y4, label='y(0)=9', color='red')
plt.plot(t5, x5, label='x(0)=10', color='blue')
plt.plot(t5, y5, label='y(0)=10', color='purple')
plt.title('Curvas componentes de la orbita superior')
plt.xlabel('Tiempo (Dias)')
plt.ylabel('Tropas')
plt.legend()
plt.grid()

# Plotting y vs. x
plt.subplot(1, 2, 2)
plt.plot(x1, y1, label='y(0)=6', color='blue')
plt.plot(x2, y2, label='y(0)=7', color='orange')
plt.plot(x3, y3, label='y(0)=8', color='green')
plt.plot(x4, y4, label='y(0)=9', color='red')
plt.plot(x5, y5, label='y(0)=10', color='purple')
plt.title('Arcos de hiperbolas (Y vs X)')
plt.xlabel('Variable X (Tropas)')
plt.ylabel('Variable Y (Tropas)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
