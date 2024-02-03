import logging
import threading

from colorama import Fore
from fastapi import FastAPI
import config, system_utils
import uvicorn, warnings
from process import Process
from config import energy_logger
from log_filters import EndpointFilter

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
    # asyncio.create_task(system_utils.computation_start(config.process))


@app.get("/computation-end/")
async def computation_end():
    system_utils.computation_end(config.process)


@app.get("/start-transmission/")
async def start_transmission():
    system_utils.start_transmission(config.process)


@app.get("/end-transmission/{bits}")
async def end_transmission(bits):
    system_utils.end_transmission(config.process, int(bits))


@app.get("/energy/")
async def energy():
    energy_logger.info(
        Fore.GREEN + f"conputation: {config.process.comp_time}, trasmission: {config.process.transmission_time}")
    comp = system_utils.estimate_computation_energy(config.process)
    tr = system_utils.estimate_communication_energy(config, config.process)
    energy_logger.info(
        Fore.MAGENTA + f"energy-conputation: {comp}, energy-trasmission: {tr}")
    ene = comp + tr
    config.process.comp_time = 0
    config.process.transmission_time = 0
    return ene


@app.get("/get-cpu-utilization/{pid}")
async def get_cpu_utilization(pid):
    return system_utils.get_cpu_u(pid)


# uvconfig = uvicorn.Config(app, host="0.0.0.0", port=8023, log_level="critical")
# server = uvicorn.Server(uvconfig)
# server.run()
# logging.critical("energy estimation service started on port "+str(8023))
uvicorn.run(app, host="0.0.0.0", port=8023)
