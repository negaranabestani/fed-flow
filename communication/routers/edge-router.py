import warnings

import uvicorn
from fastapi import FastAPI

warnings.filterwarnings('ignore')

app = FastAPI()


# Add filter to the logger


@app.get("/init/global/weights/sever")
async def getGlobalWeightFromServer():
    pass


@app.get("/local/weights/client")
async def getLocalWeightFromClient():
    pass


@app.get("/local/activation/client")
async def getLocalActivationFromClient():
    pass


@app.get("/split/layer/server")
async def getSplitLayerFromServer():
    pass


@app.get("/gradient/server")
async def getGradientFromServer():
    pass


@app.get("/gradient/edge")
async def getGradientFromEdge():
    pass


# uvconfig = uvicorn.Config(app, host="0.0.0.0", port=8023, log_level="critical")
# server = uvicorn.Server(uvconfig)
# server.run()
# logging.critical("energy estimation service started on port "+str(8023))
uvicorn.run(app, host="0.0.0.0", port=8023)
