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

# Funciones para las EDs

def cooling_eq(t, T):
    k = 0.1  # Constante de enfriamiento
    T_a = 20  # Temperatura ambiente
    return -k * (T - T_a)

def harmonic_oscillator(t, x):
    omega = 2 * np.pi  # Frecuencia angular
    return -omega**2 * x

def rc_circuit(t, V):
    Rs = 1  # Resistencia
    Cs = 1  # Capacitancia
    v1, v2 = V
    dV1dt = (1/(Rs * Cs)) * (1 - v1)
    dV2dt = (1/(Rs * Cs)) * (v1 - v2)
    return np.array([dV1dt, dV2dt])

def volterra_model(N, t):
    r1, r2, alpha, beta = 0.1, 0.1, 0.02, 0.01
    N1, N2 = N
    dN1dt = r1 * N1 * (1 - N1 / 10) - alpha * N1 * N2
    dN2dt = beta * N1 * N2 - r2 * N2
    return np.array([dN1dt, dN2dt])

# Valores de tiempo y pasos
t0 = 0  # Tiempo inicial
tf = 10  # Tiempo final
steps = 100  # Número de pasos

# Soluciones con método de Euler
# Ecuación de Enfriamiento de Newton
y0_enfriamiento = 100  # Temperatura inicial
t_enfriamiento_euler, T_enfriamiento_euler = euler_method(cooling_eq, y0_enfriamiento, t0, tf, steps)

# Oscilador Armónico
y0_oscillator = 1  # Posición inicial
t_oscillator_euler, x_oscillator_euler = euler_method(harmonic_oscillator, y0_oscillator, t0, tf, steps)

# Circuito RC
V0 = np.array([0, 0])  # Condiciones iniciales
t_rc_euler = np.linspace(t0, tf, steps)
V_rc_euler = np.zeros((steps, 2))
V_rc_euler[0] = V0

for i in range(1, steps):
    dt = t_rc_euler[i] - t_rc_euler[i - 1]
    V_rc_euler[i] = V_rc_euler[i - 1] + rc_circuit(t_rc_euler[i - 1], V_rc_euler[i - 1]) * dt

# Modelo de Volterra No Lineal
N0 = np.array([40, 9])  # Población inicial de presas y depredadores
t_volterra_euler = np.linspace(t0, tf, steps)
N_volterra_euler = np.zeros((steps, 2))
N_volterra_euler[0] = N0

for i in range(1, steps):
    dt = t_volterra_euler[i] - t_volterra_euler[i - 1]
    N_volterra_euler[i] = N_volterra_euler[i - 1] + volterra_model(N_volterra_euler[i - 1], t_volterra_euler[i - 1]) * dt

# Soluciones con método RK2
# Ecuación de Enfriamiento de Newton
t_enfriamiento_rk2, T_enfriamiento_rk2 = rk2_method(cooling_eq, y0_enfriamiento, t0, tf, steps)

# Oscilador Armónico
t_oscillator_rk2, x_oscillator_rk2 = rk2_method(harmonic_oscillator, y0_oscillator, t0, tf, steps)

# Circuito RC
V_rc_rk2 = np.zeros((steps, 2))
V_rc_rk2[0] = V0

for i in range(1, steps):
    dt = t_rc_euler[i] - t_rc_euler[i - 1]
    k1 = rc_circuit(t_rc_euler[i-1], V_rc_rk2[i-1])
    k2 = rc_circuit(t_rc_euler[i-1] + dt, V_rc_rk2[i-1] + k1 * dt)
    V_rc_rk2[i] = V_rc_rk2[i-1] + 0.5 * (k1 + k2) * dt

# Modelo de Volterra No Lineal
N_volterra_rk2 = np.zeros((steps, 2))
N_volterra_rk2[0] = N0

for i in range(1, steps):
    dt = t_volterra_euler[i] - t_volterra_euler[i - 1]
    k1 = volterra_model(N_volterra_rk2[i - 1], t_volterra_euler[i - 1])
    k2 = volterra_model(N_volterra_rk2[i - 1] + k1 * dt, t_volterra_euler[i - 1] + dt)
    N_volterra_rk2[i] = N_volterra_rk2[i - 1] + 0.5 * (k1 + k2) * dt

# Visualización de Resultados
plt.figure(figsize=(12, 10))

# Gráfico Ecuación de Enfriamiento de Newton
plt.subplot(2, 2, 1)
plt.plot(t_enfriamiento_euler, T_enfriamiento_euler, label='Euler', color='blue')
plt.plot(t_enfriamiento_rk2, T_enfriamiento_rk2, label='RK2', color='orange', linestyle='--')
plt.title('Ecuación de Enfriamiento de Newton')
plt.xlabel('Tiempo')
plt.ylabel('Temperatura T(t)')
plt.axhline(y=20, color='r', linestyle='--', label='Temp. Ambiente (T_a)')
plt.legend()

# Gráfico Oscilador Armónico
plt.subplot(2, 2, 2)
plt.plot(t_oscillator_euler, x_oscillator_euler, label='Euler', color='blue')
plt.plot(t_oscillator_rk2, x_oscillator_rk2, label='RK2', color='orange', linestyle='--')
plt.title('Oscilador Armónico')
plt.xlabel('Tiempo')
plt.ylabel('Posición x(t)')
plt.legend()

# Gráfico Circuito RC
plt.subplot(2, 2, 3)
plt.plot(t_rc_euler, V_rc_euler[:, 0], label='v1 (Euler)', color='blue')
plt.plot(t_rc_euler, V_rc_rk2[:, 0], label='v1 (RK2)', color='orange', linestyle='--')
plt.plot(t_rc_euler, V_rc_euler[:, 1], label='v2 (Euler)', color='green')
plt.plot(t_rc_euler, V_rc_rk2[:, 1], label='v2 (RK2)', color='red', linestyle='--')
plt.title('Circuito RC')
plt.xlabel('Tiempo')
plt.ylabel('Voltaje')
plt.legend()

# Gráfico del Modelo de Volterra
plt.subplot(2, 2, 4)
plt.plot(t_volterra_euler, N_volterra_euler[:, 0], label='Presas (Euler)', color='blue')
plt.plot(t_volterra_euler, N_volterra_rk2[:, 0], label='Presas (RK2)', color='orange', linestyle='--')
plt.plot(t_volterra_euler, N_volterra_euler[:, 1], label='Depredadores (Euler)', color='green')
plt.plot(t_volterra_euler, N_volterra_rk2[:, 1], label='Depredadores (RK2)', color='red', linestyle='--')
plt.title('Modelo de Volterra: Presas y Depredadores')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()

plt.tight_layout()
plt.show()

# Resultados
# Crear DataFrames para mostrar resultados
data_enfriamiento = pd.DataFrame({'Tiempo': t_enfriamiento_euler, 'Temperatura (Euler)': T_enfriamiento_euler, 'Temperatura (RK2)': T_enfriamiento_rk2})
data_oscillator = pd.DataFrame({'Tiempo': t_oscillator_euler, 'Posición (Euler)': x_oscillator_euler, 'Posición (RK2)': x_oscillator_rk2})
data_rc = pd.DataFrame({'Tiempo': t_rc_euler, 'V1 (Euler)': V_rc_euler[:, 0], 'V1 (RK2)': V_rc_rk2[:, 0], 'V2 (Euler)': V_rc_euler[:, 1], 'V2 (RK2)': V_rc_rk2[:, 1]})
data_volterra = pd.DataFrame({'Tiempo': t_volterra_euler, 'Población Presas (Euler)': N_volterra_euler[:, 0], 'Población Presas (RK2)': N_volterra_rk2[:, 0], 'Población Depredadores (Euler)': N_volterra_euler[:, 1], 'Población Depredadores (RK2)': N_volterra_rk2[:, 1]})

# Mostrar las tablas de resultados
print("Resultados Ecuación de Enfriamiento de Newton:")
print(data_enfriamiento.head())

print("\nResultados Oscilador Armónico:")
print(data_oscillator.head())

print("\nResultados Circuito RC:")
print(data_rc.head())

print("\nResultados Modelo de Volterra:")
print(data_volterra.head())

