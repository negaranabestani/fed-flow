import logging
import os
import time
import requests, warnings

warnings.filterwarnings('ignore')

URL = "http://127.0.0.1:8023"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("requests").setLevel(logging.WARNING)
energy_logger = logging.getLogger(__name__)


def computation_start():
    session = requests.session()
    session.trust_env = False
    session.get(url=URL + "/computation-start/")


def computation_end():
    session = requests.session()
    session.trust_env = False
    session.get(url=URL + "/computation-end/")


def start_transmission():
    session = requests.session()
    session.trust_env = False
    session.get(url=URL + "/start-transmission/")


def end_transmission():
    session = requests.session()
    session.trust_env = False
    session.get(url=URL + "/end-transmission/")


def energy():
    session = requests.session()
    session.trust_env = False
    result = session.get(url=URL + "/energy/")
    return result.text


def pcpuc():
    session = requests.session()
    session.trust_env = False
    result = session.get(url=URL + "/")
    return result.text


time.sleep(2)

session = requests.session()
session.trust_env = False
session.get(url=URL + "/init/" + str(os.getpid()) + "/")
computation_start()
print("started")
sum = 0
for i in range(1000000000):
    # if i % 100000 == 0:
    #     energy_logger.info(f"result: {sum}")
    sum += i

computation_end()
# print(pcpuc())
# start_transmission()
# end_transmission()
print(energy())
