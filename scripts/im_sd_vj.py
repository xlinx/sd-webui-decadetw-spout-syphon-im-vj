import base64
import io
import logging
import os
import enum
import pprint
import subprocess
import time

import gradio
import numpy as np
from OpenGL import GL
from PIL import Image

import SpoutGL
import gradio as gr
from openai import OpenAI, OpenAIError

from modules import scripts, script_callbacks
from modules.processing import StableDiffusionProcessingTxt2Img
# from modules.script_callbacks import on_ui_tabs
from modules.shared import opts

log = logging.getLogger("[auto-llm]")
# log.setLevel(logging.INFO)
# Logging
log_file = os.path.join(scripts.basedir(), "auto-llm.log")

random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5'  # ⇅
restore_progress_symbol = '\U0001F300'  # 🌀
detect_image_size_symbol = '\U0001F4D0'  # 📐


# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
class EnumCmdReturnType(enum.Enum):
    JUST_CALL = 'just-call'
    LLM_USER_PROMPT = 'LLM-USER-PROMPT'
    LLM_VISION_IMG_PATH = 'LLM-VISION-IMG_PATH'
    SENDER_NAME = 'IM_SD_VJ'

    @classmethod
    def values(cls):
        return [e.value for e in cls]


def gr_img_to_pil(grimage: gradio.Image):
    return grimage.value.Image.value.fromarray(
        np.clip(255. * grimage.value.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))



def call_SpoutGL_sender(on_image_saved_params):
    global SENDER_NAME
    up_3_level_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    if on_image_saved_params is not None:
        for ele in on_image_saved_params:
            # image_path = os.path.join(base_folder, "..", "..", "..", ele.filename)
            image_path = os.path.join(up_3_level_path, ele.filename)
            pil_image = Image.open(image_path)
            SEND_WIDTH = pil_image.width
            SEND_HEIGHT = pil_image.height
            with SpoutGL.SpoutSender() as sender:
                sender.setSenderName('IM_SD_VJ')
                #for i in range(33 * 3):
                    # print("[][IM-VJ][while True]")
                    # Generating bytes in Python is very slow; ideally you should pass in a buffer obtained elsewhere
                    # or re-use an already allocated array instead of allocating one on the fly
                    # pixels = bytes(islice(cycle([randcolor(), randcolor(), randcolor(), 255]), SEND_WIDTH * SEND_HEIGHT * 4))
                    # pixels = image_to_byte_array(pil_image)
                    #bytesarray = bytes(Image.fromarray(bytesarray.reshape((SEND_WIDTH, SEND_HEIGHT, 3))).tobytes())
                    # sender.sendImage(pixels, SEND_WIDTH, SEND_HEIGHT, GL.GL_RGBA, False, 0)
                result = sender.sendImage(pil_image.tobytes("raw"), SEND_WIDTH, SEND_HEIGHT, GL.GL_RGB, False, 0)
                    # Indicate that a frame is ready to read
                sender.setFrameSync('IM_SD_VJ')
                    # Wait for next send attempt
                    #time.sleep(1. / 30)
        on_image_saved_params=[]
    return "", "",


def image_to_byte_array(image: Image) -> bytes:
    # BytesIO is a file-like buffer stored in memory
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    image.save(imgByteArr, format='PNG')
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


# def tensor_to_pil(image:Image):
#     return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# def image_to_base64(image):
#     pli_image = tensor_to_pil(image)
#     image_data = io.BytesIO()
#     pli_image.save(image_data, format='PNG', pnginfo=None)
#     image_data_bytes = image_data.getvalue()
#     encoded_image = "data:image/png;base64," + base64.b64encode(image_data_bytes).decode('utf-8')
#     # log.warning("[][image_to_base64][]"+encoded_image)
#     # if not str(base64_image).startswith("data:image"):
#     #     base64_image = f"data:image/jpeg;base64,{base64_image}"
#     return encoded_image

def _get_effective_prompt(prompts: list[str], prompt: str) -> str:
    return prompts[0] if prompts else prompt


class AutoLLM(scripts.Script):

    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "SD-Im-VJ"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def do_subprocess_action(self, llm_post_action_cmd):
        if llm_post_action_cmd.__len__() <= 0:
            return ""
        p = subprocess.Popen(llm_post_action_cmd.split(" "), text=True, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        (out, err) = p.communicate()
        ret = p.wait()
        ret = True if ret == 0 else False
        if ret:
            log.warning("Command succeeded. " + llm_post_action_cmd + " output=" + out)
            self.llm_history_array.append(["[O]PostAction-Command succeeded.", err, llm_post_action_cmd, out])
        else:
            log.warning("Command failed. " + llm_post_action_cmd + " err=" + err)
            self.llm_history_array.append(["[X]PostAction-Command failed.", err, llm_post_action_cmd, out])
        return out

    def ui(self, is_img2img):

        with gr.Blocks():
            # gr.Markdown("Blocks")
            with gr.Accordion(open=False, label="Auto-Share-OpenGL-Im-VJ 20240828"):
                with gr.Tab("Spout(win)"):
                    gr.Markdown("* Tx - Open ur third software receive SD-Image\n"
                                "* Rx - Auto Send to Inpainting\n"
                                "* You can enable both Rx-Tx receive Image then output to ur Live show Control software.\n")
                    enable_spout = gr.Checkbox(label=" Enable Spout", value=False)

                    with gr.Row():
                        enable_spout_tx = gr.Checkbox(label=" Enable_spout_Tx🌀", value=False)
                        spout_tx_fps = gr.Slider(
                            elem_id="llm_top_k_vision", label="spout_tx_fps", value=30, minimum=1, maximum=60,
                            step=1, interactive=True, hint='spout_tx_fps')

                    with gr.Row():
                        enable_spout_rx = gr.Checkbox(label=" Enable_spout_Rx🌀", value=False)
                        spout_tx_fps = gr.Slider(
                            elem_id="llm_top_k_vision", label="spout_tx_fps", value=30, minimum=1, maximum=60,
                            step=1, interactive=True, hint='spout_tx_fps')
                with gr.Tab("Syphon(mac)"):
                    gr.Markdown("* Tx - Open ur third software receive SD-Image\n"
                                "* Rx - Auto Send to Inpainting\n"
                                "* You can enable both Rx-Tx receive Image then output to ur Live show Control software.\n")
                    enable_syphon = gr.Checkbox(label=" Enable syphon", value=False)
                    with gr.Row():
                        enable_syphon_tx = gr.Checkbox(label=" Enable_syphon_Tx🌀", value=False)
                        syphon_tx_fps = gr.Slider(
                            elem_id="llm_top_k_vision", label="syphon_tx_fps", value=30, minimum=1, maximum=60,
                            step=1, interactive=True, hint='syphon_tx_fps')

                    with gr.Row():
                        enable_syphon_rx = gr.Checkbox(label=" Enable_syphon_Rx🌀", value=False)
                        syphon_tx_fps = gr.Slider(
                            elem_id="llm_top_k_vision", label="syphon_tx_fps", value=30, minimum=1, maximum=60,
                            step=1, interactive=True, hint='spout_tx_fps')

                with gr.Tab("Setup"):
                    gr.Markdown("* API-URI: LMStudio=>http://localhost:1234/v1 \n"
                                "* API-URI: ollama  => http://localhost:11434/v1 \n"
                                "* API-ModelName: LMStudio can be empty here is fine; select it LMStudio App; ollama should set like: llama3.1 (cmd:ollama list)\n"
                                "* OLLAMA OpenAI compatibility https://ollama.com/blog/openai-compatibility\n"
                                )

        return [enable_spout, enable_spout_tx, enable_spout_rx,
                enable_syphon, enable_syphon_tx, enable_syphon_rx
                ]

    # def process(self, p: StableDiffusionProcessingTxt2Img,*args):

    def postprocess(self, p: StableDiffusionProcessingTxt2Img, *args):
        global args_dict
        args_dict = dict(zip(args_keys, args))
        # if llm_is_enabled:
        if args_dict.get('enable_spout'):
            log.warning("[][][enable_spout]")
            #call_SpoutGL_sender(*args)
        # if llm_is_open_eye:
        if args_dict.get('enable_syphon'):
            log.warning("[][][enable_syphon]")

        return p.all_prompts[0]


args_dict = None
args_keys = ['enable_spout', 'enable_spout_tx', 'enable_spout_rx',
             'enable_syphon', 'enable_syphon_tx', 'enable_syphon_rx']
on_image_saved_params = []


def on_image_saved(params):
    global on_image_saved_params
    on_image_saved_params.append(params)
    call_SpoutGL_sender(on_image_saved_params)
    # log.warning(f"[event][on_image_saved][params]: {on_image_saved_params} {print_obj_x(on_image_saved_params)} {params} {print_obj_x(params)}")

script_callbacks.on_image_saved(on_image_saved)

# with gr.Row():
#    js_neg_prompt_js = gr.Textbox(label="[Negative prompt-JS]", lines=3, value="{}")
#    js_neg_result = gr.Textbox(label="[Negative prompt-JS-Result]", lines=3, value="result")
#    # self.p.change(self.process, inputs=js_result, outputs=js_result)
# with gr.Row():
#     llm_models = gr.Dropdown(
#         ['noDetect'].append(List_LLM_Models), value=['noDetect'], multiselect=False,
#         label="List LLM "
#               "Models",
#         info="get models from local LLM. (:LM Studio)"
#     )

# script_callbacks.on_ui_tabs(on_ui_tabs )
#https://platform.openai.com/docs/api-reference/introduction
