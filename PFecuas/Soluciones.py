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
# ED de Primer Orden: Ecuación de Enfriamiento de Newton
def cooling_eq(t, T):
    k = 0.1  # Constante de enfriamiento
    T_a = 20  # Temperatura ambiente
    return -k * (T - T_a)

# ED de Segundo Orden: Oscilador Armónico 
def harmonic_oscillator(t, x):
    omega = 2 * np.pi  # Frecuencia angular
    return -omega**2 * x

# Sistema de Ecuaciones de 2x2 Lineales: Circuito RC
def rc_circuit(t, V):
    Rs = 1  # Resistencia
    Cs = 1  # Capacitancia
    v1, v2 = V
    dV1dt = (1/(Rs * Cs)) * (1 - v1)
    dV2dt = (1/(Rs * Cs)) * (v1 - v2)
    return np.array([dV1dt, dV2dt])

# Sistema de Ecuaciones de 2x2 No Lineal: Modelo de Volterra
def volterra_model(N, t):
    r1, r2, alpha, beta = 0.1, 0.1, 0.02, 0.01
    N1, N2 = N
    dN1dt = r1 * N1 * (1 - N1 / 10) - alpha * N1 * N2
    dN2dt = beta * N1 * N2 - r2 * N2
    return np.array([dN1dt, dN2dt])

# Parámetros
t0 = 0  # Tiempo inicial
tf = 10  # Tiempo final
steps = 100  # Número de pasos

# Solucionando Ecuación de Enfriamiento de Newton
y0_enfriamiento = 100  # Temperatura inicial
t_enfriamiento, T_enfriamiento = euler_method(cooling_eq, y0_enfriamiento, t0, tf, steps)

# Solucionando el Oscilador Armónico
y0_oscillator = 1  # Posición inicial
t_oscillator, x_oscillator = euler_method(harmonic_oscillator, y0_oscillator, t0, tf, steps)

# Solucionando el Circuito RC
V0 = np.array([0, 0])  # Condiciones iniciales
t_rc = np.linspace(t0, tf, steps)
V_rc_euler = np.zeros((steps, 2))
V_rc_euler[0] = V0

for i in range(1, steps):
    dt = t_rc[i] - t_rc[i - 1]
    V_rc_euler[i] = V_rc_euler[i - 1] + rc_circuit(t_rc[i - 1], V_rc_euler[i - 1]) * dt

# Solucionando el Modelo de Volterra No Lineal
N0 = np.array([40, 9])  # Población inicial de presas y depredadores
t_volterra = np.linspace(t0, tf, steps)
N_volterra = np.zeros((steps, 2))
N_volterra[0] = N0

for i in range(1, steps):
    dt = t_volterra[i] - t_volterra[i - 1]
    N_volterra[i] = N_volterra[i - 1] + volterra_model(N_volterra[i - 1], t_volterra[i - 1]) * dt

# Visualización de Resultados
plt.figure(figsize=(12, 10))

# Gráfico Ecuación de Enfriamiento de Newton
plt.subplot(2, 2, 1)
plt.plot(t_enfriamiento, T_enfriamiento, label='Método de Euler')
plt.title('Ecuación de Enfriamiento de Newton')
plt.xlabel('Tiempo')
plt.ylabel('Temperatura T(t)')
plt.axhline(y=20, color='r', linestyle='--', label='Temp. Ambiente (T_a)')
plt.legend()

# Gráfico Oscilador Armónico
plt.subplot(2, 2, 2)
plt.plot(t_oscillator, x_oscillator, label='Método de Euler')
plt.title('Oscilador Armónico')
plt.xlabel('Tiempo')
plt.ylabel('Posición x(t)')
plt.legend()

# Gráfico Circuito RC
plt.subplot(2, 2, 3)
plt.plot(t_rc, V_rc_euler[:, 0], label='v1 (Voltaje en resistencia)')
plt.plot(t_rc, V_rc_euler[:, 1], label='v2 (Voltaje en capacitor)')
plt.title('Circuito RC')
plt.xlabel('Tiempo')
plt.ylabel('Voltaje')
plt.legend()

# Gráfico del Modelo de Volterra
plt.subplot(2, 2, 4)
plt.plot(t_volterra, N_volterra[:, 0], label='Población de Presas')
plt.plot(t_volterra, N_volterra[:, 1], label='Población de Depredadores')
plt.title('Modelo de Volterra: Presas y Depredadores')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()

plt.tight_layout()
plt.show()

# Resultados
# Crear DataFrames para mostrar resultados
data_enfriamiento = pd.DataFrame({'Tiempo': t_enfriamiento, 'Temperatura': T_enfriamiento})
data_oscillator = pd.DataFrame({'Tiempo': t_oscillator, 'Posición': x_oscillator})
data_rc = pd.DataFrame({'Tiempo': t_rc, 'V1': V_rc_euler[:, 0], 'V2': V_rc_euler[:, 1]})
data_volterra = pd.DataFrame({'Tiempo': t_volterra, 'Población Presas': N_volterra[:, 0], 'Población Depredadores': N_volterra[:, 1]})

# Mostrar las tablas de resultados
print("Resultados Ecuación de Enfriamiento de Newton:")
print(data_enfriamiento.head())

print("\nResultados Oscilador Armónico:")
print(data_oscillator.head())

print("\nResultados Circuito RC:")
print(data_rc.head())

print("\nResultados Modelo de Volterra:")
print(data_volterra.head())
