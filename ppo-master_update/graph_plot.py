import matplotlib.pyplot as plt
import pandas as pd
import os

my_res_path = os.path.abspath('res/1kts/')
my_xls_path = os.path.abspath('res/')

val = 0.5 # value of p

# total cost
xls_file = 'p='+str(val)+'/avg_total_p='+str(val)+'_.xlsx'

try:
    plot_list = [[] for _ in range(6)]
    i = 0
    rng = 0
    with open(os.path.join(my_xls_path, xls_file), 'rb') as f:
        xls_data = pd.read_excel(f)
        for i in range(6):
            plot_list[i] = xls_data['y_' + str(i + 1)].tolist()[:1000]
            # print(plot_list[i])
            rng = len(plot_list[i])
            # rng = 
        i += 1
    df = pd.DataFrame(
        {'x': range(rng), 'y_1': plot_list[0], 'y_2': plot_list[1], 'y_3': plot_list[2], 'y_4': plot_list[3],
         'y_5': plot_list[4], 'y_6' : plot_list[5]})
    plt.xlabel("Time Slot")
    plt.ylabel("Time Average Cost")
    plt.plot('x', 'y_1', data=df, marker='o', markevery=int(rng / 10), color='red', linewidth=1, label="ppo")
    plt.plot('x', 'y_2', data=df, marker='^', markevery=int(rng / 10), color='olive', linewidth=1, label="random")
    plt.plot('x', 'y_3', data=df, marker='s', markevery=int(rng / 10), color='cyan', linewidth=1, label="myopic")
    plt.plot('x', 'y_4', data=df, marker='*', markevery=int(rng / 10), color='skyblue', linewidth=1,
             label="fixed 0.4kW")
    plt.plot('x', 'y_5', data=df, marker='+', markevery=int(rng / 10), color='navy', linewidth=1, label="fixed 1kW")
    plt.plot('x', 'y_6', data=df, marker='x', markevery=int(rng / 10), color='green', linewidth=1, label="q learning")
    plt.legend(fancybox=True, shadow=True)
    plt.grid()
    fig_file = 'p=' + str(val) + '/avg_total_p=' + str(val) + '_.png'
    plt.savefig(os.path.join(my_res_path, fig_file))
    plt.show()
except IOError:
    print('Fuck')

# time cost
xls_file = 'p='+str(val)+'/avg_time_p='+str(val)+'_.xlsx'

try:
    plot_list = [[] for _ in range(6)]
    i = 0
    rng = 0
    with open(os.path.join(my_xls_path, xls_file), 'rb') as f:
        xls_data = pd.read_excel(f)
        for i in range(6):
            plot_list[i] = xls_data['y_' + str(i + 1)].tolist()[:1000]
            rng = len(plot_list[i])
        i += 1
    df = pd.DataFrame(
        {'x': range(rng), 'y_1': plot_list[0], 'y_2': plot_list[1], 'y_3': plot_list[2], 'y_4': plot_list[3],
         'y_5': plot_list[4], 'y_6' : plot_list[5]})
    plt.xlabel("Time Slot")
    plt.ylabel("Time Average Delay Cost")
    plt.plot('x', 'y_1', data=df, marker='o', markevery=int(rng / 10), color='red', linewidth=1, label="ppo")
    plt.plot('x', 'y_2', data=df, marker='^', markevery=int(rng / 10), color='olive', linewidth=1, label="random")
    plt.plot('x', 'y_3', data=df, marker='s', markevery=int(rng / 10), color='cyan', linewidth=1, label="myopic")
    plt.plot('x', 'y_4', data=df, marker='*', markevery=int(rng / 10), color='skyblue', linewidth=1,
             label="fixed 0.4kW")
    plt.plot('x', 'y_5', data=df, marker='+', markevery=int(rng / 10), color='navy', linewidth=1, label="fixed 1kW")
    plt.plot('x', 'y_6', data=df, marker='x', markevery=int(rng / 10), color='green', linewidth=1, label="q learning")
    plt.legend(fancybox=True, shadow=True)
    plt.grid()
    fig_file = 'p=' + str(val) + '/avg_time_p=' + str(val) + '_.png'
    plt.savefig(os.path.join(my_res_path, fig_file))
    plt.show()
except IOError:
    print('Fuck')

# back-up cost
xls_file = 'p='+str(val)+'/avg_backup_p='+str(val)+'_.xlsx'

try:
    plot_list = [[] for _ in range(6)]
    i = 0
    rng = 0
    with open(os.path.join(my_xls_path, xls_file), 'rb') as f:
        xls_data = pd.read_excel(f)
        for i in range(6):
            plot_list[i] = xls_data['y_' + str(i + 1)].tolist()[:1000]
            rng = len(plot_list[i])
        i += 1
    df = pd.DataFrame(
        {'x': range(rng), 'y_1': plot_list[0], 'y_2': plot_list[1], 'y_3': plot_list[2], 'y_4': plot_list[3],
         'y_5': plot_list[4], 'y_6' : plot_list[5]})
    plt.xlabel("Time Slot")
    plt.ylabel("Time Average Back-up Power Cost")
    plt.plot('x', 'y_1', data=df, marker='o', markevery=int(rng / 10), color='red', linewidth=1, label="ppo")
    plt.plot('x', 'y_2', data=df, marker='^', markevery=int(rng / 10), color='olive', linewidth=1, label="random")
    plt.plot('x', 'y_3', data=df, marker='s', markevery=int(rng / 10), color='cyan', linewidth=1, label="myopic")
    plt.plot('x', 'y_4', data=df, marker='*', markevery=int(rng / 10), color='skyblue', linewidth=1,
             label="fixed 0.4kW")
    plt.plot('x', 'y_5', data=df, marker='+', markevery=int(rng / 10), color='navy', linewidth=1, label="fixed 1kW")
    plt.plot('x', 'y_6', data=df, marker='x', markevery=int(rng / 10), color='green', linewidth=1, label="q learning")
    plt.legend(fancybox=True, shadow=True)
    plt.grid()
    fig_file = 'p=' + str(val) + '/avg_backup_p=' + str(val) + '_.png'
    plt.savefig(os.path.join(my_res_path, fig_file))
    plt.show()
except IOError:
    print('Fuck')

# battery cost
xls_file = 'p='+str(val)+'/avg_battery_p='+str(val)+'_.xlsx'

try:
    plot_list = [[] for _ in range(6)]
    i = 0
    rng = 0
    with open(os.path.join(my_xls_path, xls_file), 'rb') as f:
        xls_data = pd.read_excel(f)
        for i in range(6):
            plot_list[i] = xls_data['y_' + str(i + 1)].tolist()[:1000]
            rng = len(plot_list[i])
        i += 1
    df = pd.DataFrame(
        {'x': range(rng), 'y_1': plot_list[0], 'y_2': plot_list[1], 'y_3': plot_list[2], 'y_4': plot_list[3],
         'y_5': plot_list[4], 'y_6' : plot_list[5]})
    plt.xlabel("Time Slot")
    plt.ylabel("Time Average Battery Power Cost")
    plt.plot('x', 'y_1', data=df, marker='o', markevery=int(rng / 10), color='red', linewidth=1, label="ppo")
    plt.plot('x', 'y_2', data=df, marker='^', markevery=int(rng / 10), color='olive', linewidth=1, label="random")
    plt.plot('x', 'y_3', data=df, marker='s', markevery=int(rng / 10), color='cyan', linewidth=1, label="myopic")
    plt.plot('x', 'y_4', data=df, marker='*', markevery=int(rng / 10), color='skyblue', linewidth=1,
             label="fixed 0.4kW")
    plt.plot('x', 'y_5', data=df, marker='+', markevery=int(rng / 10), color='navy', linewidth=1, label="fixed 1kW")
    plt.plot('x', 'y_6', data=df, marker='x', markevery=int(rng / 10), color='green', linewidth=1, label="q learning")
    plt.legend(fancybox=True, shadow=True)
    plt.grid()
    fig_file = 'p=' + str(val) + '/avg_battery_p=' + str(val) + '_.png'
    plt.savefig(os.path.join(my_res_path, fig_file))
    plt.show()
except IOError:
    print('Fuck')

# energy cost
xls_file = 'p='+str(val)+'/avg_energy_p='+str(val)+'_.xlsx'

try:
    plot_list = [[] for _ in range(6)]
    i = 0
    rng = 0
    with open(os.path.join(my_xls_path, xls_file), 'rb') as f:
        xls_data = pd.read_excel(f)
        for i in range(6):
            plot_list[i] = xls_data['y_' + str(i + 1)].tolist()[:1000]
            rng = len(plot_list[i])
        i += 1
    df = pd.DataFrame(
        {'x': range(rng), 'y_1': plot_list[0], 'y_2': plot_list[1], 'y_3': plot_list[2], 'y_4': plot_list[3],
         'y_5': plot_list[4], 'y_6' : plot_list[5]})
    plt.xlabel("Time Slot")
    plt.ylabel("Time Average Energy Cost")
    plt.plot('x', 'y_1', data=df, marker='o', markevery=int(rng / 10), color='red', linewidth=1, label="ppo")
    plt.plot('x', 'y_2', data=df, marker='^', markevery=int(rng / 10), color='olive', linewidth=1, label="random")
    plt.plot('x', 'y_3', data=df, marker='s', markevery=int(rng / 10), color='cyan', linewidth=1, label="myopic")
    plt.plot('x', 'y_4', data=df, marker='*', markevery=int(rng / 10), color='skyblue', linewidth=1,
             label="fixed 0.4kW")
    plt.plot('x', 'y_5', data=df, marker='+', markevery=int(rng / 10), color='navy', linewidth=1, label="fixed 1kW")
    plt.plot('x', 'y_6', data=df, marker='x', markevery=int(rng / 10), color='green', linewidth=1, label="q learning")
    plt.legend(fancybox=True, shadow=True)
    plt.grid()
    fig_file = 'p=' + str(val) + '/avg_energy_p=' + str(val) + '_.png'
    plt.savefig(os.path.join(my_res_path, fig_file))
    plt.show()
except IOError:
    print('Fuck')

# ppo area chart
xls_file_names = ['p='+str(val)+'/avg_time_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_backup_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_battery_p='+str(val)+'_.xlsx']
plot_list = [[] for _ in range(3)]
i = 0
rng = 0
for name in xls_file_names:
    try:
        with open(os.path.join(my_xls_path, name), 'rb') as f:
            xls_data = pd.read_excel(f)
            plot_list[i] = xls_data['y_1'].tolist()[:1000]
            rng = len(plot_list[i])
            i += 1
    except IOError:
        print('Fuck')

xx = range(rng)
fig = plt.stackplot(xx, plot_list, edgecolor = 'black', labels = ['Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '+++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('PPO')
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.), ncol=3, fancybox=True, shadow=True)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
my_file = 'p='+str(val)+'/ppo_area'+'p='+str(val)+'.png'
plt.savefig(os.path.join(my_res_path, my_file))
plt.show()

# random area chart
xls_file_names = ['p='+str(val)+'/avg_time_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_backup_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_battery_p='+str(val)+'_.xlsx']
plot_list = [[] for _ in range(3)]
i = 0
rng = 0
for name in xls_file_names:
    try:
        with open(os.path.join(my_xls_path, name), 'rb') as f:
            xls_data = pd.read_excel(f)
            plot_list[i] = xls_data['y_2'].tolist()[:1000]
            rng = len(plot_list[i])
            i += 1
    except IOError:
        print('Fuck')

xx = range(rng)
fig = plt.stackplot(xx, plot_list, edgecolor = 'black', labels = ['Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '+++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('Random')
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.), ncol=3, fancybox=True, shadow=True)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
my_file = 'p='+str(val)+'/random_area'+'p='+str(val)+'.png'
plt.savefig(os.path.join(my_res_path, my_file))
plt.show()

# myopic area chart
xls_file_names = ['p='+str(val)+'/avg_time_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_backup_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_battery_p='+str(val)+'_.xlsx']
plot_list = [[] for _ in range(3)]
i = 0
rng = 0
for name in xls_file_names:
    try:
        with open(os.path.join(my_xls_path, name), 'rb') as f:
            xls_data = pd.read_excel(f)
            plot_list[i] = xls_data['y_3'].tolist()[:1000]
            rng = len(plot_list[i])
            i += 1
    except IOError:
        print('Fuck')

xx = range(rng)
fig = plt.stackplot(xx, plot_list, edgecolor = 'black', labels = ['Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '+++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('Myopic')
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.), ncol=3, fancybox=True, shadow=True)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
my_file = 'p='+str(val)+'/myopic_area'+'p='+str(val)+'.png'
plt.savefig(os.path.join(my_res_path, my_file))
plt.show()

# fixed 1 area chart
xls_file_names = ['p='+str(val)+'/avg_time_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_backup_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_battery_p='+str(val)+'_.xlsx']
plot_list = [[] for _ in range(3)]
i = 0
rng = 0
for name in xls_file_names:
    try:
        with open(os.path.join(my_xls_path, name), 'rb') as f:
            xls_data = pd.read_excel(f)
            plot_list[i] = xls_data['y_4'].tolist()[:1000]
            rng = len(plot_list[i])
            i += 1
    except IOError:
        print('Fuck')

xx = range(rng)
fig = plt.stackplot(xx, plot_list, edgecolor = 'black', labels = ['Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '+++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('Fixed 0.4kW')
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.), ncol=3, fancybox=True, shadow=True)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
my_file = 'p='+str(val)+'/fixed_0.4kW_area'+'p='+str(val)+'.png'
plt.savefig(os.path.join(my_res_path, my_file))
plt.show()

# fixed 2 area chart
xls_file_names = ['p='+str(val)+'/avg_time_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_backup_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_battery_p='+str(val)+'_.xlsx']
plot_list = [[] for _ in range(3)]
i = 0
rng = 0
for name in xls_file_names:
    try:
        with open(os.path.join(my_xls_path, name), 'rb') as f:
            xls_data = pd.read_excel(f)
            plot_list[i] = xls_data['y_5'].tolist()[:1000]
            rng = len(plot_list[i])
            i += 1
    except IOError:
        print('Fuck')

xx = range(rng)
fig = plt.stackplot(xx, plot_list, edgecolor = 'black', labels = ['Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '+++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('Fixed 1kW')
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.), ncol=3, fancybox=True, shadow=True)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
my_file = 'p='+str(val)+'/fixed_1kW_area'+'p='+str(val)+'.png'
plt.savefig(os.path.join(my_res_path, my_file))
plt.show()

# dqn area chart
xls_file_names = ['p='+str(val)+'/avg_time_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_backup_p='+str(val)+'_.xlsx', 'p='+str(val)+'/avg_battery_p='+str(val)+'_.xlsx']
plot_list = [[] for _ in range(3)]
i = 0
rng = 0
for name in xls_file_names:
    try:
        with open(os.path.join(my_xls_path, name), 'rb') as f:
            xls_data = pd.read_excel(f)
            plot_list[i] = xls_data['y_6'].tolist()[:1000]
            rng = len(plot_list[i])
            i += 1
    except IOError:
        print('Fuck')

xx = range(rng)
fig = plt.stackplot(xx, plot_list, edgecolor = 'black', labels = ['Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '+++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('DQN')
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.), ncol=3, fancybox=True, shadow=True)
# lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
my_file = 'p='+str(val)+'/dqn_area'+'p='+str(val)+'.png'
plt.savefig(os.path.join(my_res_path, my_file))
plt.show()