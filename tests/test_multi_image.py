

import os
from predict import FluxDevKontextPredictor
from tests.test_util import get_params


predictor = FluxDevKontextPredictor()
predictor.setup()

def test_style_transfer():
    input_image = "tests/resources/lady.png"
    context_images = ["tests/resources/sample/img_0.png", "tests/resources/sample/img_1.png", "tests/resources/sample/img_2.png"]
    input_images = [input_image]
    params = get_params(predictor, {})
    params["prompt"] = "Generate an image of the woman in an 80s cyberpunk style"
    params["output_format"] = "png"
    params["input_images"] = input_images
    params["multi_image_mode"] = "separate"
    predictor.predict(**params)
    os.system("mv output.png ./tests/outputs/style-transfer-separate-no-context.png")

    for i in context_images:
        params["input_images"].append(i)
        if len(params["input_images"]) > 2:
            params["prompt"] = "Generate an image of the woman in the 80s cyberpunk style of the other images"
        else:
            params["prompt"] = "Generate an image of the woman in the 80scyberpunk style of the other image"

        predictor.predict(**params)
        os.system("mv output.png ./tests/outputs/style-transfer-separate-{}.png".format(len(input_images)))


def test_style_generation():
    context_images = ["tests/resources/sample/img_0.png", "tests/resources/sample/img_1.png", "tests/resources/sample/img_2.png"]
    input_images = [input_image]
    params = get_params(predictor, {})
    params["prompt"] = "Using this style, a selfie of a woman"
    params["output_format"] = "png"
    params["input_images"] = input_images
    predictor.predict(**params)
    os.system("mv output.png ./tests/outputs/style-transfer-no-context.png")

    for i in context_images:
        params["input_images"].append(i)
        if len(params["input_images"]) > 2:
            params["prompt"] = "Generate an image of the woman in the 80s cyberpunk style of the other images"
        else:
            params["prompt"] = "Generate an image of the woman in the 80scyberpunk style of the other image"

        predictor.predict(**params)
        os.system("mv output.png ./tests/outputs/style-transfer-{}.png".format(len(input_images)))


def test_composition():
    prompt = "Put the person on top of the black horse with red eyes"
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
            os.system("mv output.png ./tests/outputs/prompt-2-{}-{}.png".format(multi_image_mode, mode))

if __name__ == "__main__":
    test_style_transfer()
