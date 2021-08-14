import matplotlib.pyplot as plt
import pandas as pd
import os

my_path = os.path.abspath('res/compare_p/')
xls_file = 'compare_p.xlsx'
xls_data = pd.read_excel(os.path.join(my_path, xls_file))
# print(xls_data)
x_range = xls_data['p'].tolist()
unscaled_time_cost = xls_data['time'].tolist()
unscaled_energy_cost = xls_data['energy'].tolist()
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Coefficient')
ax1.set_xticks(x_range)
ax1.set_ylabel('Delay time Cost', color=color)
plt.ylim((9.5, 12))
ax1.plot(x_range, unscaled_time_cost, color=color, marker='o', label='Delay time Cost')
ax1.tick_params(axis='y', labelcolor=color)
plt.legend(loc='lower center', bbox_to_anchor=(0.25, 0.), fancybox=True, shadow=True)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Energy Cost', color=color)  # we already handled the x-label with ax1
ax2.plot(x_range, unscaled_energy_cost, color=color, marker='^', label = 'Energy Cost')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
my_file = 'compare_p/time-energy.png'
plt.savefig('res/compare_p/compare_p.png')
plt.grid()
plt.legend(loc='lower center', bbox_to_anchor=(0.75, 0.), fancybox=True, shadow=True)
plt.show() 