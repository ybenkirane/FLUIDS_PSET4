"""
@author: Yacine Benkirane

Collaborator: Alexandre Stuart

Code for Probelms 1.2 and 1.3
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit


#Part 1.2, plotting cost vs speed

vel_list = np.logspace(-1, 4, 100)
m_list = np.array([0.01, 0.1, 1.0, 10.0, 100.0])



def Tucker1973(bird_vel, bird_m):
    metabolic_drag = (10.6/bird_vel)*((bird_m / 0.035)**(-1/4))
    parasitic_drag = (2.25e-4)*(bird_vel**2)*((bird_m / 0.035)**(-1/3))
    induced_drag = (2.6/(bird_vel**2))*((bird_m / 0.035)**(1/3))
    return metabolic_drag + parasitic_drag + induced_drag

epsilon_cost_1 = Tucker1973(vel_list, 0.01)
epsilon_cost_2 = Tucker1973(vel_list, 0.1)
epsilon_cost_3 = Tucker1973(vel_list, 1.0)
epsilon_cost_4 = Tucker1973(vel_list, 10.0)
epsilon_cost_5 = Tucker1973(vel_list, 100.0)




plt.plot(vel_list, epsilon_cost_1, '.')
plt.plot(vel_list, epsilon_cost_2, '--')
plt.plot(vel_list, epsilon_cost_3, ':')
plt.plot(vel_list, epsilon_cost_4, '-.')
plt.plot(vel_list, epsilon_cost_5, '-')
plt.yscale("log")
plt.xscale("log")
plt.legend(['0.01 kg','0.1 kg','1 kg','10 kg','100 kg'])
plt.xlabel('Bird Velocity (m/s)')
plt.ylabel("Cost of Transport")
plt.title("Bird transport cost vs Velocity")
plt.show()


#Part 1.3

velopt_1, cost_min_1 = vel_list[np.argmin(epsilon_cost_1)], epsilon_cost_1[np.argmin(epsilon_cost_1)]
velopt_2, cost_min_2 = vel_list[np.argmin(epsilon_cost_2)], epsilon_cost_2[np.argmin(epsilon_cost_2)]
velopt_3, cost_min_3 = vel_list[np.argmin(epsilon_cost_3)], epsilon_cost_3[np.argmin(epsilon_cost_3)]
velopt_4, cost_min_4 = vel_list[np.argmin(epsilon_cost_4)], epsilon_cost_4[np.argmin(epsilon_cost_4)]
velopt_5, cost_min_5 = vel_list[np.argmin(epsilon_cost_5)], epsilon_cost_5[np.argmin(epsilon_cost_5)]



print("{:.1f}".format(velopt_1), "{:.1f}".format(velopt_2), "{:.1f}".format(velopt_3), "{:.1f}".format(velopt_4), "{:.1f}".format(velopt_5))
print("{:.3f}".format(cost_min_1), "{:.3f}".format(cost_min_2), "{:.3f}".format(cost_min_3), "{:.3f}".format(cost_min_4), "{:.3f}".format(cost_min_5))
cost_min_list = np.array([cost_min_1, cost_min_2, cost_min_3, cost_min_4, cost_min_5])



def PowLaw(mass, exp, const):
    return const*(mass**exp)



popt1, pcov1 = curve_fit(PowLaw, m_list, cost_min_list)
print(np.sqrt(pcov1))
print('const = ', popt1[0], '\n exponent = ', popt1[1])

vel_list2 = np.logspace(-2, 2, 100)



plt.plot(m_list, cost_min_list, 'b*')
plt.plot(vel_list2,PowLaw(vel_list2,popt1[0],popt1[1]), 'r')


plt.xscale("log")
plt.yscale("log")
ax = plt.gca()
ax.set_yscale('log')


plt.tick_params(axis='y', which='minor')
ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
plt.xlabel('Mass (kg)')
plt.ylabel("Minimized Transport Cost")


plt.title("Minimal Transport Cost")
plt.legend(['Simulation','Best Linear Fit'])
plt.show()