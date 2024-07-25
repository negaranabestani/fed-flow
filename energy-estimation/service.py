import logging
import subprocess
import threading
import warnings

import uvicorn
from colorama import Fore
from fastapi import FastAPI

import config
import system_utils
from config import energy_logger
from log_filters import EndpointFilter
from process import Process

warnings.filterwarnings('ignore')

app = FastAPI()
config.process = Process(0)

excluded_endpoints = ["/init/", "/", "/computation-start/", "/computation-end/", "/start-transmission/",
                      "/end-transmission/", "/get-cpu-utilization/"]

# Add filter to the logger
logging.getLogger("uvicorn.access").addFilter(EndpointFilter(excluded_endpoints))


@app.get("/init/{pid}/{simnet}")
async def init(pid, simnet):
    system_utils.init_p(config.process, pid, simnet)


@app.get("/")
async def root():
    return system_utils.pcpuc(config.process, config)


@app.get("/computation-start/")
async def computation_start():
    x = threading.Thread(target=system_utils.computation_start, args=(config.process,))
    x.start()
    # energy_logger.info("computation started")
    # asyncio.create_task(system_utils.computation_start(config.process))


@app.get("/computation-end/")
async def computation_end():
    system_utils.computation_end(config.process)


@app.get("/start-transmission/")
async def start_transmission():
    system_utils.start_transmission(config.process)


@app.get("/end-transmission/{bits}")
async def end_transmission(bits):
    # energy_logger.info(Fore.RED + f"{int(bits)}")
    system_utils.end_transmission(config.process, int(bits))


@app.get("/energy/")
async def energy():
    energy_logger.info(
        Fore.GREEN + f"conputation: {config.process.comp_time}, trasmission: {config.process.transmission_time}" +
        Fore.RESET)
    comp = system_utils.estimate_computation_energy(config.process)
    tr = system_utils.estimate_communication_energy(config, config.process)
    energy_logger.info(
        Fore.MAGENTA + f"energy-conputation: {comp}, energy-trasmission: {tr}" + Fore.RESET)
    ene = comp + tr

    cores = int(subprocess.run("nproc", capture_output=True, shell=True, text=True).stdout)
    # energy_logger.info(f"cpus: {cores}")
    print(f"config.process.cpu_u_count : {config.process.cpu_u_count}")
    print(f"cores : {cores}")

    utilization = config.process.cpu_utilization / config.process.cpu_u_count / 100 / cores

    config.process.comp_time = 0
    config.process.cpu_u_count = 0
    config.process.end_comp = False
    config.process.cpu_utilization = 0
    config.process.transmission_time = 0
    return utilization


@app.get("/energy/time/comp_tr")
async def energy_and_time_comp_tr():
    comp_time = config.process.comp_time
    tr_time = config.process.transmission_time
    energy_logger.info(Fore.GREEN + f"computation: {comp_time}, transmission: {tr_time}" + Fore.RESET)

    comp_e = system_utils.estimate_computation_energy(config.process)
    tr_e = system_utils.estimate_communication_energy(config, config.process)
    energy_logger.info(Fore.MAGENTA + f"energy-computation: {comp_e}, energy-transmission: {tr_e}" + Fore.RESET)

    comp_tr = str(comp_e) + "," + str(tr_e) + "," + str(comp_time) + "," + str(tr_time)

    config.process.comp_time = 0
    config.process.cpu_u_count = 0
    config.process.end_comp = False
    config.process.cpu_utilization = 0
    config.process.transmission_time = 0
    return comp_tr


@app.get("/get-cpu-utilization/{pid}")
async def get_cpu_utilization(pid):
    return system_utils.get_cpu_u(pid)


# uvconfig = uvicorn.Config(app, host="0.0.0.0", port=8023, log_level="critical")
# server = uvicorn.Server(uvconfig)
# server.run()
# logging.critical("energy estimation service started on port "+str(8023))
uvicorn.run(app, host="0.0.0.0", port=8023)
