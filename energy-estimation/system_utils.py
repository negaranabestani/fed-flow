import logging
import random
import subprocess
import time
import config
from config import energy_logger


def init_p(process, pid, simulate_network):
    process.pid = pid
    config.simulate_network = simulate_network


def estimate_computation_energy(process):
    if process.cpu_u_count == 0:
        return 0
    max_energy_per_core = ((process.system_energy / 1000) / process.cpu_u_count)
    utilization = process.cpu_utilization / process.cpu_u_count
    computation_time = process.comp_time
    return max_energy_per_core * utilization * computation_time


def estimate_communication_energy(config, process):
    return process.transmission_time * config.tx_power


def start_transmission(process):
    process.start_tr_time = time.time()


def end_transmission(process, bits):
    process.end_tr_time = time.time()
    if config.simulate_network == True:
        # energy_logger.info(f"simnet:{config.simulate_network}")
        b = bits / (process.end_tr_time - process.start_tr_time)
        b = random.uniform(0.6 * b, 1.2 * b)
        process.transmission_time += bits / b
    else:
        # b = bits / (process.end_tr_time - process.start_tr_time)
        # energy_logger.info((f"bandwidth: {b}, {bits}"))
        # # b *= 0.01
        # process.transmission_time += bits / b
        process.transmission_time += (process.end_tr_time - process.start_tr_time)


def get_cpu_u(pid):
    # print("pid" + str(pid))
    data = subprocess.run("top -n 1 -b -p " + str(pid), capture_output=True, shell=True, text=True)
    # print("result"+str(data.stdout))
    result = data.stdout.split("\n")
    target = result[7].split(" ")
    # print(target[len(target)-8])
    return float(target[len(target) - 8])


def get_power_now():
    # data = subprocess.run("cat /sys/class/power_supply/BAT0/power_now", capture_output=True, shell=True, text=True)
    # print("power"+data.stdout)
    # return data.stdout
    return config.power


def get_TX():
    pass


def computation_start(process):
    process.start_comp_time = time.time()
    while not process.end_comp:
        process.cpu_u_count += 1
        process.cpu_utilization += get_cpu_u(process.pid)
        process.system_energy += float(get_power_now())


def pcpuc(process, config):
    return process.cpu_u_count


def computation_end(process):
    if process is None:
        logging.error("no started process")
        return
    process.end_comp = True
    process.end_comp_time = time.time()
    process.comp_time += process.end_comp_time - process.start_comp_time
