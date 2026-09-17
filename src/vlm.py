import sys
import re

# Windows compatibility workaround:
# Windows Application Control is blocking the native
# regex DLL used by Transformers.
sys.modules["regex"] = re

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers.image_utils import load_image


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading VLM...")


processor = AutoProcessor.from_pretrained(
    MODEL_ID
)


model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
).to(DEVICE)


print("VLM loaded successfully!")


def describe_image(image_path, question):

    image = load_image(
        image_path
    )


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "path": image_path
                },
                {
                    "type": "text",
                    "text": question
                },
            ],
        }
    ]


    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


    inputs = inputs.to(
        DEVICE
    )


    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200
    )


    input_length = inputs[
        "input_ids"
    ].shape[1]


    generated_text = processor.batch_decode(
        generated_ids[:, input_length:],
        skip_special_tokens=True,
    )[0]


    return generated_text.strip()


if __name__ == "__main__":

    image_path = "assets/test.jpg"


    result = describe_image(
        image_path,
        "Describe this image in detail."
    )


    print(
        "\n===== VLM RESPONSE ====="
    )


    print(result)