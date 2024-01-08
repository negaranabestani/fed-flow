import warnings
from app.config import config

import requests

URL = "http://127.0.0.1:8023"
warnings.filterwarnings('ignore')

def init(pid):
    session = requests.session()
    session.trust_env = False
    session.get(url=URL + "/init/" + str(pid) + "/" + str(config.simnet))


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


def end_transmission(bits):
    session = requests.session()
    session.trust_env = False
    session.get(url=URL + "/end-transmission/" + str(bits))


def energy():
    session = requests.session()
    session.trust_env = False
    result = session.get(url=URL + "/energy/")
    return result.text
