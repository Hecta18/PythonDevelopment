import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the model
def model(Y, t):
    # TODO use x , t values
    x, y = Y
    dxdt = -0.1 * y
    dydt = -0.064 * y
    return [dxdt, dydt]

# Initial conditions
Y0_1 = [10, 6]
Y0_2 = [10, 7]
Y0_3 = [10, 8]
Y0_4 = [10, 9]
Y0_5 = [10, 10]

# Time points
t = np.linspace(0, 10, 100)

# Solve ODEs
sol1 = odeint(model, Y0_1, t)
sol2 = odeint(model, Y0_2, t)
sol3 = odeint(model, Y0_3, t)
sol4 = odeint(model, Y0_4, t)
sol5 = odeint(model, Y0_5, t)

# Extracting solutions
x1, y1 = sol1.T
x2, y2 = sol2.T
x3, y3 = sol3.T
x4, y4 = sol4.T
x5, y5 = sol5.T

plt.figure(figsize=(12, 6))

# Plotting the curves
plt.subplot(1, 2, 1)
plt.plot(t, y1, label='y(0)=6', color='blue')
plt.plot(t, y2, label='y(0)=7', color='orange')
plt.plot(t, y3, label='y(0)=8', color='green')
plt.plot(t, y4, label='y(0)=9', color='red')
plt.plot(t, y5, label='y(0)=10', color='purple')
plt.title('Lanchester Hypothetical Arches')
plt.xlabel('Time (units of force)')
plt.ylabel('Variable Y (units)')
plt.legend()
plt.grid()

# Plotting y vs. x
plt.subplot(1, 2, 2)
plt.plot(x1, y1, label='y(0)=6', color='blue')
plt.plot(x2, y2, label='y(0)=7', color='orange')
plt.plot(x3, y3, label='y(0)=8', color='green')
plt.plot(x4, y4, label='y(0)=9', color='red')
plt.plot(x5, y5, label='y(0)=10', color='purple')
plt.title('Phase Plane (Y vs X)')
plt.xlabel('Variable X (units)')
plt.ylabel('Variable Y (units)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
