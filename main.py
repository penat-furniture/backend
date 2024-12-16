import os
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import utils

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory="images"), name="images")


@app.get("/")
async def root():
    return {"status": True}


@app.get("/api/")
async def root():
    return {"api": True}


@app.post("/api/suggest/")
async def suggest(data: Optional[dict] = None):
    image = data.get("image", None)
    print("GOT IMAGE:", image)
    step = data.get("step", "1")
    room = data.get("room")
    if image == '[object Object]':
        print("Мать ебал")
        image = None
    limits = data.get("limits", None)
    images = utils.get_images(room, step, image, limits)
    return {
        "images": images,
    }


@app.get("/api/limits/")
async def limits():
    return utils.get_limits()
