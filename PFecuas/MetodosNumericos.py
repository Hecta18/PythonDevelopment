import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Método de Euler
def euler_method(f, y0, t0, tf, steps):
    t = np.linspace(t0, tf, steps)
    y = np.zeros(steps)
    y[0] = y0
    dt = t[1] - t[0]
    
    for i in range(1, steps):
        y[i] = y[i-1] + f(t[i-1], y[i-1]) * dt
        
    return t, y

# Método de Runge-Kutta de segundo orden (RK2)
def rk2_method(f, y0, t0, tf, steps):
    t = np.linspace(t0, tf, steps)
    y = np.zeros(steps)
    y[0] = y0
    dt = t[1] - t[0]
    
    for i in range(1, steps):
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + dt, y[i-1] + k1 * dt)
        y[i] = y[i-1] + (k1 + k2) * dt / 2
        
    return t, y

# Función ejemplo: dy/dt = -2y (solución: y(t) = y0 * exp(-2t))
def example_function(t, y):
    return -2 * y

# Parámetros
y0 = 1  # Condición inicial
t0 = 0  # Tiempo inicial
tf = 5  # Tiempo final
steps = 100  # Número de pasos

# Implementación
t_euler, y_euler = euler_method(example_function, y0, t0, tf, steps)
t_rk2, y_rk2 = rk2_method(example_function, y0, t0, tf, steps)

# Visualización de resultados
plt.plot(t_euler, y_euler, label='Euler')
plt.plot(t_rk2, y_rk2, label='RK2')
plt.title('Métodos Numéricos')
plt.xlabel('Tiempo')
plt.ylabel('y(t)')
plt.legend()
plt.show()


# Resultados
# Datos de resultados para Euler y RK2
data = {
    'Tiempo': t_euler,
    'Euler': y_euler,
    'RK2': y_rk2
}

# Crear DataFrame
results_df = pd.DataFrame(data)

# Mostrar la tabla de resultados
print(results_df.head())  # Muestra las primeras filas

# Gráfico comparativo
plt.figure(figsize=(10, 6))

plt.plot(t_euler, y_euler, label='Método de Euler', color='blue')
plt.plot(t_rk2, y_rk2, label='Método RK2', color='orange')
# TODO Add non-linear solution calculation
plt.scatter(solution[0], solution[1], color='red', label='Solución No Lineal', s=100)

plt.title('Comparación de Métodos Numéricos y Solución No Lineal')
plt.xlabel('Tiempo')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

# Resultado del sistema no lineal
non_linear_data = {
    'Variable': ['x', 'y'],
    # TODO Add non-linear solution calculation
    'Valor': [solution[0], solution[1]]
}

non_linear_df = pd.DataFrame(non_linear_data)
print(non_linear_df)
