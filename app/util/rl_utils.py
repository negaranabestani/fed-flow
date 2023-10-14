import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from tensorforce import Environment
from app.RL_training.customEnv import CustomEnvironment
from app.model.entity.rl_model import NoSplitting, TRPO, AC, TensorforceAgent, RandomAgent


def draw_graph(figSizeX, figSizeY, x, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    # Create a plot
    plt.figure(figsize=(int(figSizeX), int(figSizeY)))  # Set the figure size
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, pictureName))
    plt.close()
    # plt.show()


def draw_hist(x, title, xlabel, savePath, pictureName, saveFig=True):
    # Create a plot
    plt.hist(x, 10)
    plt.title(title)
    plt.xlabel(xlabel)
    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, pictureName))
    plt.close()
    # plt.show()


def draw_scatter(x, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, pictureName))
    plt.close()
    # plt.show()


def draw_3dGraph(x, y, z, xlabel, ylabel, zlabel):
    fig = go.Figure(data=[go.Mesh3d(x=x,
                                    y=y,
                                    z=z,
                                    opacity=0.7, )])

    fig.update_layout(scene=dict(xaxis_title=xlabel,
                                 yaxis_title=ylabel,
                                 zaxis_title=zlabel,
                                 xaxis_showspikes=False,
                                 yaxis_showspikes=False))

    fig.show()


def sigmoidActivation(x: float) -> float:
    """ It returns 1/(1+exp(-x)). where the values lies between zero and one """

    return 1 / (1 + np.exp(-x))


def tanhActivation(x: float) -> float:
    """ It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1."""

    return np.tanh(x)


def normalizeReward(maxAmount, minAmount, x):
    P = [maxAmount, 0]
    Q = [minAmount, 1]
    lineGradient = (P[1] - Q[1]) / (P[0] - Q[0])
    y = lineGradient * (x - Q[0]) + Q[1]
    return y


def convert_To_Len_th_base(n, arr, modelLen, deviceNumber, allPossible):
    a: str = ""
    for i in range(deviceNumber * 2):
        a += str(arr[n % modelLen])
        n //= modelLen
    allPossible.append(a)


def allPossibleSplitting(modelLen, deviceNumber):
    arr = [i for i in range(0, modelLen + 1)]
    allPossible = list()
    for i in range(pow(modelLen, deviceNumber * 2)):
        # Convert i to Len th base
        convert_To_Len_th_base(i, arr, modelLen, deviceNumber, allPossible)
    result = list()
    for item in allPossible:
        isOk = True
        for j in range(0, len(item) - 1, 2):
            if int(item[j]) > int(item[j + 1]):
                isOk = False
        if isOk:
            result.append(item)
    return result


def createEnv(timestepNum, iotDeviceNum, edgeDeviceNum, fraction, rewardTuningParams) -> Environment:
    return Environment.create(
        environment=CustomEnvironment(rewardTuningParams=rewardTuningParams, iotDeviceNum=iotDeviceNum,
                                      edgeDeviceNum=edgeDeviceNum, fraction=fraction),
        max_episode_timesteps=timestepNum)


def createAgent(agentType, fraction, timestepNum, environment, saveSummariesPath):
    if agentType == 'ac':
        return AC.create(fraction=fraction, environment=environment, timestepNum=timestepNum,
                         saveSummariesPath=saveSummariesPath)
    elif agentType == 'tensorforce':
        return TensorforceAgent.create(fraction=fraction, environment=environment,
                                       timestepNum=timestepNum, saveSummariesPath=saveSummariesPath)
    elif agentType == 'trpo':
        return TRPO.create(fraction=fraction, environment=environment,
                           timestepNum=timestepNum, saveSummariesPath=saveSummariesPath)
    elif agentType == 'random':
        return RandomAgent.RandomAgent(environment=environment)
    elif agentType == 'noSplitting':
        return NoSplitting.NoSplitting(environment=environment)
    else:
        raise Exception('Invalid config select from [ppo, ac, tensorforce, random]')
