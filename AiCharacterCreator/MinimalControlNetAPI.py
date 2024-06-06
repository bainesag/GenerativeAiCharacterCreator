import io
import cv2
import base64
import requests
from matplotlib import pyplot as plt

from PIL import Image


# A1111 URL
url = "http://127.0.0.1:7860"

# Read Image in RGB order
img = cv2.imread('InputFiles/HeartPose1.png')
# Encode into PNG and send to ControlNet
retval, bytes = cv2.imencode('.png', img)
encoded_image = base64.b64encode(bytes).decode('utf-8')

# A1111 payload
payload = {
    "prompt": 'jim lee,  <lora:animetarotV51:1>,  1girl,  <lora:jim_lee_offset_right_filesize:0.65>, fully clothed',
    "negative_prompt": "badhandv4 , easynegative, text sci-fi, leotard",
    "batch_size": 1,
    "steps": 30,
    "cfg_scale": 7,
    "width":512,
    "height":896,
    "alwayson_scripts": {
        "controlnet": {
            "args": [
                {
                    "input_image": encoded_image,
                    "model": "control_v11p_sd15_openpose [cab727d4]",
                    "pixel_perfect":True
                }
            ]
        }
    }
}

response = requests.get(url=f'{url}/controlnet/model_list', json=payload)

response = requests.get(url=f'{url}/controlnet/module_list', json=payload)

# Trigger Generation
response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

# Read results
r = response.json()
result = r['images'][0]
image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
image.save('outputs/output.png')