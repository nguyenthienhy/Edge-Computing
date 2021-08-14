import math
import numpy as np

def get_delay_local(number_of_server, local_workload, server_service_rate):
    if number_of_server == 0:
        return math.inf
    if local_workload < 0:
        return 0
    return local_workload / (number_of_server * server_service_rate - local_workload)

def cal(action, total_workload, congestion_rate):
    opt_val = math.inf
    ans = [-1, -1]
    for number_of_server in range(1, 11):
        coeff = [1, 1, 1 - action / number_of_server]
        y = np.roots(coeff)
        x = np.roots(coeff)[1]
        print('roots = ' + str(y))
        local_workload = number_of_server * x
        if local_workload <= 0:
            continue
        print('# servers ' + str(number_of_server))
        print('Local workload ' + str(local_workload))
        print('Cost delay ' + str((get_delay_local(number_of_server, local_workload, 20)
                                + (total_workload - local_workload) * congestion_rate)))
        if opt_val > (get_delay_local(number_of_server, local_workload, 20)
                        + (total_workload - local_workload) * congestion_rate):
            ans = [number_of_server, local_workload]
            opt_val = get_delay_local(number_of_server, local_workload, 20) + \
                        (total_workload - local_workload) * congestion_rate
    return ans

print('Ans ' + str(cal(200, 400, 10)))