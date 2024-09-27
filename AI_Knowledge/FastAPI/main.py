from fastapi import FastAPI
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/")
async def root():
    return {"message": "Hello World"}

# http://127.0.0.1:8000/items/3 -> Works because FastAPI converts "3" to int
# http://127.0.0.1:8000/items/foo -> Fails because "foo" cannot be converted to int
# http://127.0.0.1:8000/items/4.2 -> Fials because "4.2" cannot be converted to int, only to float
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

"""
You could need the parameter to contain /home/johndoe/myfile.txt, with a leading slash (/).
In that case, the URL would be: /files//home/johndoe/myfile.txt, with a double slash (//) between files and home.
"""
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}

@app.get("/query_items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]

