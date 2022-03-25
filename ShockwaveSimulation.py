# -*- coding: utf-8 -*-
"""
@author: Yacine Benkirane

Collaborators: Alexandre
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio as imo

N_list = 100
StepCount = 800
dt = 0.01
dx = 2.0


m1 = np.ones(N_list)        
m2 = np.zeros(N_list)      
m3 = np.ones(N_list)     

x = np.arange(N_list) * dx  
u = np.zeros(N_list+1)     

pressure = np.zeros(N_list)
sound_sp = np.zeros(N_list)

Amp, sigma = 10000, N_list/12
m3 = m3 + Amp * np.exp(-(x - x.max()/2) ** 2 / sigma ** 2)

def advection(f, u, dt, dx):
    J = np.zeros(len(f)+1)
    J[1:-1] = np.where(u[1:-1] > 0, f[:-1] * u[1:-1], f[1:] * u[1:-1])
    f = f - (dt / dx) * (J[1:] - J[:-1]) 
    return f

plt.ion()
fig, ax = plt.subplots(2,1)

special_density = np.zeros(N_list)
special_velocity = np.zeros(N_list)


x1, = ax[0].plot(x, m1, 'g-')
x2, = ax[1].plot(x, m2, 'r-')


ax[0].set_xlabel('Position')
ax[1].set_xlabel('Position')
ax[0].set_ylabel('Density')
ax[1].set_ylabel('Mach number $\mathcal{M}$')
ax[0].set_title('Density and Mach number of Shock')



ax[0].set_xlim([0, dx*N_list+1])
ax[0].set_ylim([0, 5])
ax[0].plot(x, np.ones(N_list)*4, 'k-.')
ax[0].legend(['Density Expectation'])
ax[1].set_xlim([0, dx*N_list+1])
ax[1].set_ylim([0, 10])




fig.canvas.draw()

for ct in range(StepCount):
    
    # Velocity Advection
    u[1:-1] = 0.5 * ((m2[:-1] / m1[:-1]) + (m2[1:] / m1[1:]))

    # Density Momentum Energy Update
    m1 = advection(m1, u, dt, dx)
    m2 = advection(m2, u, dt, dx)

    # Pressure Update
    pressure = (2/5)*(m3 - ((m2**2)/(2*m1)))
    
    # Euler Equation
    m2[1:-1] = m2[1:-1] - 0.5 * (dt / dx) * (pressure[2:] - pressure[:-2])

    # Reflective Boundary
    m2[0] = m2[0] - 0.5 * (dt / dx) * (pressure[1] - pressure[0])
    m2[-1] = m2[-1] - 0.5 * (dt / dx) * (pressure[-1] - pressure[-2])

    # Vel Advection
    u[1:-1] = 0.5 * ((m2[:-1] / m1[:-1]) + (m2[1:] / m1[1:]))
    
    # Energy Advection
    m3 = advection(m3, u, dt, dx)
    
    # Pressure Update
    pressure = (2/5)*(m3 - ((m2**2)/(2*m1)))
    
    # Energy
    m3[1:-1] = m3[1:-1] - 0.5 * (dt / dx) * ((m2[2:]/m1[2:])*pressure[2:] - (m2[:-2]/m1[:-2])*pressure[:-2])
    
    # Reflective Boundary
    m3[0] = m3[0] - 0.5 * (dt / dx) * ((m2[1]/m1[1])*pressure[1] - (m2[0]/m1[0])*pressure[0])
    m3[-1] = m3[-1] - 0.5 * (dt / dx) * ((m2[-1]/m1[-1])*pressure[-1] - (m2[-2]/m1[-2])*pressure[-2])
    
    # Pressure and Sound Update
    pressure = (2/5)*(m3 - ((m2**2)/(2*m1)))
    sound_sp = np.sqrt((5/3)*(pressure/m1))
    
    if ct == 250:
        special_density = m1
        special_velocity = m2/m1
    
    # Plot Update
    x1.set_ydata(m1)
    x2.set_ydata(np.abs(m2)/sound_sp)
    fig.canvas.draw()
    plt.pause(0.001)
    





