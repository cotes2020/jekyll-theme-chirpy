from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


# 创建一个 FastAPI「实例」
app = FastAPI()
# app 同样在如下命令中被 uvicorn 所引用：
# uvicorn main:app --reload

# https://127.0.0.1:8000/
@app.get("/")
def read_root():
    return {"Hello": "World"}


# https://127.0.0.1:8000/items/5?q=somequery.
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
    # return {"item_name": item.name, "item_id": item_id}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}
