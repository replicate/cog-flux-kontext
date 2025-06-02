

import os
from predict import FluxDevKontextPredictor
from tests.test_util import get_params


predictor = FluxDevKontextPredictor()

prompt = "Put the person on top of the horse"
input_iamges = ["tests/resources/person.jpg", "tests/resources/horse.jpg"]
reversed_images = input_iamges[::-1]

for input_images in [input_iamges, reversed_images]:
    if input_images[0] == reversed_images[0]:
        mode = "reversed"
    else:
        mode = "normal"
    params = get_params(predictor, {})
    params["prompt"] = prompt
    params["input_images"] = input_iamges
    params["output_format"] = "png"

    for multi_image_mode in ["concat", "separate"]:
        params["multi_image_mode"] = multi_image_mode
        predictor.predict(**params)
        os.system("mv output.png ./tests/outputs/out-{}-{}.png".format(multi_image_mode, mode))


