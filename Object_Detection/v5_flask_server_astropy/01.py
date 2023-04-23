import base64
from PIL import Image
import io
file = "aaa.txt"
with open(file,"rb") as f:
    data = f.read()
    print(data)
    img = Image.open(io.BytesIO(data))
    size = img.size
    print(data)