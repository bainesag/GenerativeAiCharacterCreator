import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import io
import cv2
import base64
import requests
import json
import os
import glob
import re
from matplotlib import pyplot as plt
import random
from PIL import Image

hostname  = "localhost"
serverPort = 8080
OUTPUTPATH = "AiCharacterCreator/MassImageGenerationTraitsSorted"
# A1111 URL
url = "http://127.0.0.1:7860"

# A1111 payload
BASEPAYLOAD = {
    "prompt": 'A clear high quality image of a modern ',
    "negative_prompt": " text, easynegative, naked, explicit, nude, gore, bleeding, exposed, unsafe, historical",
    "batch_size": 1,
    "steps": 15,
    "cfg_scale": 7,
    "width":512,
    "height":896,
    "alwayson_scripts": {
            "controlnet": {
            "args": [
                {
                    "input_image": "",
                    "model": "control_v11p_sd15_openpose [cab727d4]",
                    "pixel_perfect":True,
                    "control_mode":2
                }
            ]
        }

    }
}


def massGenerate():
    poses = ["poseone"]
    genders = ["male","female","nonbinary"]
    traitones = ["none"] #Can add more body shapes to this list
    traittwos = ["long", "shoulder length", "short"] #Can add more hair colours to this list ["Brown", "Black", "Red", "Coloured", "Blonde"] Removed these as assuming minimal impact with hair colour
    traitthrees = ["none"] # #Can add more eye colours here ["Blue", "Green", "Brown", "Black", "Heterochromic"] Removed these as it was an excessive number of images
    traitfours = ["feminine", "masculine"]#, "Professional", "Urban", "Fantasy", "Casual", "Sci-fi"] #Can add more clothing styles (in the form "wearing X style clothes")
    traitfives = ["none"],# "Black", "White", "Asian", "Mixed", "Middle Eastern", "Indian"] #Can add more races to this list - Worth Clarifying
    traitsixes = ["none"]#, "nurse", "beautician", "engineer", "teacher", "professor"] #Add more professions to this list (in the form "who is a X")
    seeds= []
    for i in range(0,12):
        seeds.append(random.randrange(1000000,9999999))
    
    #This is absolutely terrible but its going to take forever to process anyway so why not
    for p in poses:
        for g in genders:
            for t1 in traitones:
                for t2 in traittwos:
                    for t3 in traitthrees:
                        for t4 in traitfours:
                            for t5 in traitfives:
                                for t6 in traitsixes:
                                    characterJson = {
                                        "pose":p,
                                        "gender":g,
                                        "traits":[t1,t2,t3,t4,t5,t6]
                                    }
                                    for seed in seeds:
                                        genImage(characterJson,seed)

def genImage(buildString, seed):
    payload = buildPayload(buildString, seed)
    print(OUTPUTPATH + "/*")# + f'/{buildString["traits"][5]}/{buildString["gender"]}/*')
    fileList = glob.glob(OUTPUTPATH + f'/{buildString["gender"]}/{buildString["traits"][3]}_Clothes/*')
    if ( not (fileList == [] or fileList == None)):
        latestFile = max(fileList, key=os.path.getctime)
        print(latestFile)
        validFiles = re.search(r"[0-9]+",latestFile)
        numoutputs = int(validFiles.group())+1
    else:
        numoutputs = 1

    imgResponse = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    imgResponse = imgResponse.json()
    result = imgResponse['images'][0]


    #save pre cesored image
    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))

    tempPath = OUTPUTPATH + f'/{buildString["gender"]}/{buildString["traits"][3]}_Clothes'
    tempName = f'{buildString["gender"]}_{buildString["traits"][1]}_hair_{buildString["traits"][3]}_clothes'

    #numoutputs = int(len([name for name in os.listdir(OUTPUTPATH + "/") if os.path.isfile(os.path.join("OUTPUTPATH + /", name))])/2)
    image.save(f'{tempPath}/{tempName}_image_{numoutputs}.png')
    json_data = json.dumps(payload)
    with open(f'{tempPath}/{tempName}_prompt_{numoutputs}.txt', 'w') as text_file:
        text_file.write(payload["prompt"])
    with open(f'{tempPath}/{tempName}_fullPrompt_{numoutputs}.json', 'w') as json_file:
        json_file.write(json.dumps(payload))

    """
    #Censorship of image not present for mass image generation (Will need moderation after)

    with open(OUTPUTPATH + f'/image{numoutputs}.png', 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    censorPayload = {
        "input_image": base64_image,  # the image you wish to censor
        #"input_mask": base64_image_mask,  # optional, if used will be combined with the NudeNet mask
        "enable_nudenet": True,  # enable NudeNet detect and generate the censor mask
        "output_mask": True,  # return the generated mask
        "filter_type": "Fill color",  # the type of filter that will be used for censoring the image
        "blur_radius": 10,  # control the strength of gaussian blur when using "Variable blur" or "Gaussian blur"
        "blur_strength_curve": 3,  # control the blur strength gradient for "Variable blur"
        "pixelation_factor": 5,  # the pixelation factor when using "Pixelate"
        "fill_color": "#000000",  # the fill color when using "Fill color"
        "mask_shape": "Ellipse",  # the shape of the masked NudeNet regions
        "mask_blend_radius": 10,  # the blurring of the combined NudeNet and input_mask before censoring is applied to the input_image
        "rectangle_round_radius": 0,  # controls corner the rounding radius when mask_shape is "Rounded rectangle"
        "nms_threshold": 0.5,  # Non-Maximum Suppression threshold of the NudeNet detected regions
        # the following three list of 18 float configures which category is censored
        # the confidence threshold of each category for it to be censored
        # and the amount that each detected regions will be expanded in horizontal and vertical direction
        # each element in the list correspond to one NudeNet label, the order can be found below in Default category configuration or on webui's api "/docs" page
        # confidence thresholds float [0, 1] when set to 1 disables this category
        # this example below censors [Female_breast_exposed, Female_genitalia_exposed, Anus_exposed, Male_genitalia_exposed]
        "thresholds":           [1, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1, 1, 1, 1, 1, 0.25, 1.0, 1.0, 1],
        # expand horizontal / vertical, float [0, inf]
        "expand_horizontal":    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "expand_vertical":      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }

    response = requests.post(url=f'{url}/nudenet/censor', json=censorPayload)
    #save censored image if censorship took place
    if response.status_code == 200:
        response = response.json()
        if response['image']:
            #overwrite result with censored version
            result = response['image']
            image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
            image.save(OUTPUTPATH + f'/Output{numoutputs}Censored.png')
    """


def buildPayload(json, seed):
        #make copy of playload so base playload isn't overwritten
        prompt = BASEPAYLOAD.copy()
        
        role = json.get("pose")
        print(json)
        #prompt["seed"] = seed
        #determine pose for role
        if role != "default":
            #prompt["prompt"] += f" {role},"
            match role:
                case "poseone":
                    imgName = "APose.png"
                    #prompt["prompt"] += "An empathetic person, "
                case "posetwo":
                    imgName = "APose.png"
                    prompt["prompt"] += "A capable fighter, "
                case "posethree":
                    imgName = "BrainsPose1.png"
                    prompt["prompt"] += "A strategic genious, "
                case "posefour":
                    imgName = "LeaderPose1.png"
                    prompt["prompt"] += "A leader, "
                case "posefive":
                    imgName = "TPoseOpenPose.png"
                    prompt["prompt"] += "Someone Tposing, "
                case "posesix":
                    imgName = "YawnOpenPose.png"
                    prompt["prompt"] += "Someone yawning, "
                case _:
                    imgName = "LeaderPose1.png"

        # Read Image in RGB order
        img = cv2.imread(f'AiCharacterCreator/InputFiles/{imgName}')
        # Encode into PNG and send to ControlNet
        retval, bytes = cv2.imencode('.png', img)
        encoded_image = base64.b64encode(bytes).decode('utf-8')
        #prompt["prompt"] += "a "
        
        # if (json.get("traits")[4] != "none"):
        #     if (json.get("gender") != "None"):
        #         prompt["prompt"] += "a "
        #     if (json.get("traits")[4] == "Mixed"):
        #         prompt["prompt"] += json.get("traits")[4] + " race"
        #     else:
        #         prompt["prompt"] += json.get("traits")[4] + " "

        match json.get("gender"):
            case "male":
                prompt["prompt"] += f"male"
            case "female":
                prompt["prompt"] += f"female"
            case "nonbinary":
                prompt["prompt"] += f" androgynous person"
            case _:
                if(json.get("traits")[4] == "none"):
                    prompt["prompt"] += "person"
                pass

        # print(json.get("traits")[0])
        # if json.get("traits")[0] != "none":
        #     prompt["prompt"] += " with a " + json.get("traits")[0] + " body type"
        #if json.get("traits")[1] != "none":
        prompt["prompt"] += " with " + json.get("traits")[1] + " hair"
        # if json.get("traits")[2] != "none":
        #     prompt["prompt"] += ", " + json.get("traits")[2] + " eyes"
        # if json.get("traits")[3] != "none":
        prompt["prompt"] += " wearing (" + json.get("traits")[3] + " style clothes:1.8)"
        #prompt["prompt"] += " who is a"
        #prompt["prompt"] += " " + json.get("traits")[5]
        #for i in json.get("traits"):
        #    prompt["prompt"] += f" {i},"
        #print(prompt)
        prompt["alwayson_scripts"]["controlnet"]["args"][0]["input_image"] = encoded_image
        print(prompt["prompt"])
        return prompt


if __name__ == '__main__':
    massGenerate()