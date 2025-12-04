import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd 


#------------ Import data to fix paramters 
data = pd.read_csv('AAPL_stock.csv', index_col=0, parse_dates=True)

S=data['Price']
returns = np.log(S/S.shift(1)).dropna()

S0 = data['Price'].iloc[-1]    
mu = returns.mean()
sigma = returns.std()
T = 1          #en años
dt = 1/252     
N_simulations = 500  
K=110
r=.0413

#------------------GBM for MC simulation

def simulate_gbm(S0, mu, sigma, T, dt, N_simulations=1):
    N = int(T / dt)  # Número de pasos de tiempo
    t = np.linspace(0, T, N)  # Vector de tiempos
    S = np.zeros((N, N_simulations))
    S[0, :] = S0
    
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt), size=N_simulations)  # Incremento de Wiener
        S[i, :] = S[i-1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    
    return t, S


#------------------Black Scholes model
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# -------------Random Walk Simulation with GBM
t, S = simulate_gbm(S0, mu, sigma, T, dt, N_simulations)

plt.figure(figsize=(10, 5))
for i in range(N_simulations):
    label = f"Simulación {i+1}" if N_simulations<6 else None
    plt.plot(t, S[:, i], label=label)
plt.xlabel("Time (years)")
plt.ylabel("Asset Price")
plt.title("Asset Price Simulation using GBM")
plt.legend()
plt.grid()
plt.show()

#----------------- Calculate the mean of final values
valores_finales = S[-1, :]
maximos=np.maximum(valores_finales - K, 0)
promedio=np.mean(maximos)
call_mc = promedio* np.exp(-r * T)  # bring it to PV


# ----------------- Calculation using Black-Scholes
call_bs = black_scholes_call(S0, K, T, r, sigma)

# Resultados
print(f"Price of a call option (Monte Carlo): {call_mc:.4f}")
print(f"Price of a call option (Black-Scholes): {call_bs:.4f}")



