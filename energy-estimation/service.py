import asyncio
import os
import subprocess
import threading
from fastapi import FastAPI
import config, system_utils
import uvicorn, warnings
from process import Process

warnings.filterwarnings('ignore')

app = FastAPI()
config.process = Process(0)


@app.get("/init/{pid}")
async def init(pid):
    system_utils.init_p(config.process, pid)


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


@app.get("/end-transmission/")
async def end_transmission():
    system_utils.end_transmission(config.process)


@app.get("/energy/")
async def energy():
    return system_utils.estimate_computation_energy(config.process) + system_utils.estimate_communication_energy(config,
                                                                                                                 config.process)


@app.get("/get-cpu-utilization/{pid}")
async def get_cpu_utilization(pid):
    return system_utils.get_cpu_u(pid)


uvicorn.run(app, host="0.0.0.0", port=8023)
