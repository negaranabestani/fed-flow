import warnings

import requests

from app.config import config

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
    try:
        return float(result.text)
    except Exception:
        return 0


def energy_and_time_comp_tr():
    session = requests.session()
    session.trust_env = False
    result = session.get(url=URL + "/energy/time/comp_tr/").text.split(",")
    comp_e = float(result[0][1:])
    tr_e = float(result[1][:])
    comp_time = float(result[2][:])
    tr_time = float(result[3][:len(result[3]) - 1])
    return comp_e, tr_e, comp_time, tr_time
