import logging
import multiprocessing
import random
import subprocess
import time

import psutil
from colorama import Fore

import config
from config import energy_logger


def init_p(process, pid, simulate_network):
    process.pid = pid
    config.simulate_network = simulate_network


def estimate_computation_energy(process):
    if process.cpu_u_count == 0:
        return 0
    # max_energy_per_core = ((process.system_energy) / process.cpu_u_count)
    cores = multiprocessing.cpu_count()
    # energy_logger.info(f"cpus: {cores}")
    utilization = process.cpu_utilization / process.cpu_u_count / 100 / cores
    # energy_logger.info(Fore.RED+f"{utilization}")
    computation_time = process.comp_time
    # energy_logger.info(Fore.LIGHTYELLOW_EX + f"{process.cpu_u_count}")
    return get_power_now() * utilization * computation_time


def estimate_communication_energy(config, process):
    return process.transmission_time * config.tx_power


def start_transmission(process):
    process.start_tr_time = time.time()


def end_transmission(process, bits):
    # energy_logger.info(Fore.MAGENTA+f"{bits}")
    process.end_tr_time = time.time()
    if config.simulate_network == True:
        # energy_logger.info(f"simnet:{config.simulate_network}")
        b = bits / (process.end_tr_time - process.start_tr_time)
        b = random.uniform(0.6 * b, 1.2 * b)
        process.transmission_time += bits / b
    else:
        # b = bits / (process.end_tr_time - process.start_tr_time)
        # energy_logger.info((f"bandwidth: {bits/(process.end_tr_time - process.start_tr_time)}, {bits}"))
        # # b *= 0.01
        # process.transmission_time += bits / b
        process.transmission_time += (process.end_tr_time - process.start_tr_time)


def get_cpu_u(pid):
    try:
        pid = int(pid)
        # Create a Process object for the given PID
        process = psutil.Process(pid)
        # Get the CPU utilization (as a percentage)
        cpu_utilization = process.cpu_percent(interval=1.0)
        return cpu_utilization
    except Exception as e:
        return 0


def get_power_now():
    # data = subprocess.run("cat /sys/class/power_supply/BAT0/power_now", capture_output=True, shell=True, text=True)
    # print("power"+data.stdout)
    # return data.stdout
    return config.power


def get_TX():
    pass


def computation_start(process):
    config.process.end_comp = False
    process.start_comp_time = time.time()
    # energy_logger.info(f": {process.end_comp}")
    while not process.end_comp:
        process.cpu_u_count += 1
        # energy_logger.info(f"count: {process.cpu_u_count}")
        process.cpu_utilization += get_cpu_u(process.pid)
        # process.system_energy += float(get_power_now())


def pcpuc(process, config):
    return process.cpu_u_count


def computation_end(process):
    if process is None:
        logging.error("no started process")
        return
    process.end_comp = True
    process.end_comp_time = time.time()
    process.comp_time += process.end_comp_time - process.start_comp_time
