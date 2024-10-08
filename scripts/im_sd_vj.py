import array
import datetime
import enum
import io
import logging
import multiprocessing
import os
import platform
import subprocess
import threading
import time
from itertools import repeat

import gradio as gr
import numpy as np
from OpenGL import GL
from PIL import Image

if platform.system() == "Windows":
    import SpoutGL
elif platform.system() == "Darwin":
    import syphon
    from syphon.utils.numpy import copy_image_to_mtl_texture
    from syphon.utils.raw import create_mtl_texture

from modules import scripts, script_callbacks
from modules.processing import StableDiffusionProcessingTxt2Img

# from modules.script_callbacks import on_ui_tabs

log = logging.getLogger("[auto-VJ]")
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

up_3_level_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
threading_spout: threading = None
threading_fps: int = 30


# class VarClass(BaseModel):
#     enable_spout: bool
#     enable_spout_tx: bool
#     spout_tx_fps: int
#     enable_spout_rx: bool
#     spout_rx_fps: int
#     enable_syphon: bool
#     enable_syphon_tx: bool
#     syphon_tx_fps: int
#     enable_syphon_rx: bool
#     syphon_rx_fps: int
#     SENDER_NAME: str
#     subprocess_history_array = List[str]
# last_one_image: {}


# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
class EnumCmdReturnType(enum.Enum):
    JUST_CALL = 'just-call'
    LLM_USER_PROMPT = 'LLM-USER-PROMPT'
    LLM_VISION_IMG_PATH = 'LLM-VISION-IMG_PATH'

    @classmethod
    def values(cls):
        return [e.value for e in cls]


def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def tensor_to_pil(image: Image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def _get_effective_prompt(prompts: list[str], prompt: str) -> str:
    return prompts[0] if prompts else prompt


# def do_subprocess_action(llm_post_action_cmd):
#     if len(llm_post_action_cmd) <= 0:
#         return ""
#     p = subprocess.Popen(llm_post_action_cmd.split(" "), text=True, shell=True, stdout=subprocess.PIPE,
#                          stderr=subprocess.PIPE)
#     (out, err) = p.communicate()
#     ret = p.wait()
#     ret = True if ret == 0 else False
#     if ret:
#         log.warning("Command succeeded. " + llm_post_action_cmd + " output=" + out)
#         VarClass.subprocess_history_array.append(
#             ["[O]PostAction-Command succeeded.", err, llm_post_action_cmd, out])
#     else:
#         log.warning("Command failed. " + llm_post_action_cmd + " err=" + err)
#         VarClass.subprocess_history_array.append(["[X]PostAction-Command failed.", err, llm_post_action_cmd, out])
#     return out
class ImageFileX:
    def __init__(self, fname):
        self.filename = fname


class IM_SD_VJ(scripts.Script):
    last_one_image = None
    opened_image_path: str = ''
    threading_stop_flag: bool = False
    threading_all: threading = None
    SENDER_NAME = 'SD-VJ'
    pil_image = None

    if platform.system() == "Windows":
        sender = SpoutGL.SpoutSender()
        sender.setSenderName(SENDER_NAME)
    elif platform.system() == "Darwin":
        sender = syphon.SyphonMetalServer("Demo")
        texture = create_mtl_texture(sender.device, 512, 512)

    def __init__(self) -> None:
        super().__init__()
        self.spout_image_rx = None
        self.spout_image_tx = None
        self.all_args_key = []
        self.all_args_val = []
        self.all_args_dict = {}

        # self.last_one_image={}

    def call_Syphon_rx(self):
        return ""

    def call_Syphon_tx(self):
        # create server and texture
        # create texture data
        texture_data = np.zeros((512, 512, 4), dtype=np.uint8)
        texture_data[:, :, 0] = 255  # fill red
        texture_data[:, :, 3] = 255  # fill alpha
        # copy texture data to texture and publish frame
        copy_image_to_mtl_texture(texture_data, self.texture)
        self.sender.publish_frame_texture(self.texture)

    def call_Spout_tx(self):
        if (self.last_one_image is not None) and (hasattr(self.last_one_image, 'filename')) and (
                type(self.last_one_image.filename) == str):
            image_path = os.path.join(up_3_level_path, self.last_one_image.filename)
            if self.opened_image_path != self.last_one_image.filename:
                self.opened_image_path = self.last_one_image.filename
                self.pil_image = Image.open(image_path)
                # self.spout_image_tx.value = image_path
        if self.pil_image is not None:
            result = self.sender.sendImage(self.pil_image.tobytes("raw"), self.pil_image.width, self.pil_image.height,
                                           GL.GL_RGB, False, 0)
            #result = self.sender.sendTexture(  sendTextureID, GL_TEXTURE_2D, self.pil_image.width, self.pil_image.height, True, 0)
            self.sender.setFrameSync(self.SENDER_NAME)

    def call_Spout_rx(self):
        with SpoutGL.SpoutReceiver() as receiver:
            receiver.setReceiverName("SpoutGL-test")
            buffer = None
            while True:
                result = receiver.receiveImage(buffer, GL.GL_RGBA, False, 0)
                if receiver.isUpdated():
                    width = receiver.getSenderWidth()
                    height = receiver.getSenderHeight()
                    buffer = array.array('B', repeat(0, width * height * 4))
                if buffer and result and not SpoutGL.helpers.isBufferEmpty(buffer):
                    print("Got bytes", bytes(buffer[0:64]), "...")
                # Wait until the next frame is ready
                # Wait time is in milliseconds; note that 0 will return immediately
                receiver.waitFrameSync(self.SENDER_NAME, 10000)

    def send_pick_image(self, pick):
        log.warning(pick)
        if type(pick) == str:
            self.last_one_image = ImageFileX(pick)

    def threading_cancel(self):
        self.threading_stop_flag = True

    def threading_run(self, spout_tx_enable, spout_rx_enable, spout_fps,
                      syphon_tx_enable, syphon_rx_enable, syphon_fps):
        if platform.system() == "Windows":
            log.warning(f"[][Win][]")
            while True:
                # log.warning(f"[{str(datetime.datetime.now())}][threading_run][spout_fps]{spout_fps}")
                if spout_tx_enable:
                    self.call_Spout_tx()
                if spout_rx_enable:
                    self.call_Spout_rx()
                time.sleep(1. / spout_fps)
                if self.threading_stop_flag:
                    break
        elif platform.system() == "Darwin":
            log.warning(f"[][Mac][]")
            while True:
                # log.warning(f"[{str(datetime.datetime.now())}][threading_run][spout_fps]{spout_fps}")
                if syphon_tx_enable:
                    self.call_Spout_tx()
                if syphon_rx_enable:
                    self.call_Spout_rx()
                time.sleep(1. / syphon_fps)
                if self.threading_stop_flag:
                    break
        elif platform.system() == "Linux":
            log.warning(f"[][Linux not support][]")

    def threading_start(self, spout_tx_enable, spout_rx_enable, spout_fps,
                        syphon_tx_enable, syphon_rx_enable, syphon_fps):
        self.threading_stop_flag = True

        self.threading_all = threading.Thread(target=self.threading_run,
                                              args=(spout_tx_enable, spout_rx_enable, spout_fps,
                                                    syphon_tx_enable, syphon_rx_enable, syphon_fps))
        self.threading_stop_flag = False
        self.threading_all.start()

    def title(self):
        return "SD-Im-VJ"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):

        with gr.Blocks():
            # gr.Markdown("Blocks")
            with gr.Accordion(open=False, label="Auto GPU Sharing SD-Result v20240828"):
                with gr.Tab("Spout(win)"):
                    gr.Markdown("* Tx - Open ur third software receive SD-Image\n"
                                "* Rx - Auto Send to Inpainting\n"
                                "* You can enable both Rx-Tx receive Image then output to ur Live show Control software.\n"
                                "* Enable then Click Start button to monitor SD-Image\n"
                                "* Using OpenGL GPU share memory sync between SD-Img and 3rd software\n")

                    with gr.Row():
                        spout_tx_enable = gr.Checkbox(label=" Enable_spout_Tx🌀", value=False)
                        spout_rx_enable = gr.Checkbox(label=" Enable_spout_Rx🌀", value=False)
                    spout_fps = gr.Slider(
                        label="Tx/Rx spout_tx_rx_fps", value=30, minimum=1, maximum=60,
                        step=1, interactive=True, hint='spout_tx_fps')
                    # with gr.Row():
                    #     self.spout_image_tx = gr.Image(label="2. Pick to send [spout_image_tx]",
                    #                                    type="filepath", visable=False)
                    #     self.spout_image_rx = gr.Image(label="2. [spout_image_rx]",
                    #                                    type="filepath", visable=False)

                    with gr.Row():
                        spout_timer_start = gr.Button("[Tx] Start (monitor SD-Img)")
                        spout_timer_cancel = gr.Button("[Tx] Stop Send")

                with gr.Tab("Syphon(mac)"):
                    gr.Markdown("* Tx - Open ur third software receive SD-Image\n"
                                "* Rx - Auto Send to Inpainting\n"
                                "* You can enable both Rx-Tx receive Image then output to ur Live show Control software.\n")
                    with gr.Row():
                        syphon_tx_enable = gr.Checkbox(label=" Enable_syphon_Tx🌀", value=False)
                        syphon_rx_enable = gr.Checkbox(label=" Enable_syphon_Rx🌀", value=False)
                    syphon_fps = gr.Slider(
                        label="syphon_tx_fps", value=30, minimum=1, maximum=60,
                        step=1, interactive=True, hint='syphon_tx_fps')
                    with gr.Row():
                        syphon_timer_start = gr.Button("[Tx] Start Threading")
                        syphon_timer_cancel = gr.Button("[Tx] Stop Threading")

                with gr.Tab("Setup"):
                    gr.Markdown("* API-URI: LMStudio=>http://localhost:1234/v1 \n"
                                "* API-URI: ollama  => http://localhost:11434/v1 \n"
                                "* API-ModelName: LMStudio can be empty here is fine; select it LMStudio App; ollama should set like: llama3.1 (cmd:ollama list)\n"
                                "* OLLAMA OpenAI compatibility https://ollama.com/blog/openai-compatibility\n"
                                )

        self.all_args_val = [spout_tx_enable, spout_rx_enable, spout_fps,
                             syphon_tx_enable, syphon_rx_enable, syphon_fps
                             ]
        # self.all_args_key = ['spout_tx_enable', 'spout_rx_enable', 'spout_fps',
        #                      'syphon_tx_enable', 'syphon_rx_enable', 'syphon_fps'
        #                      ]
        # all_args_dict = dict(zip(self.all_args_key, self.all_args_val))
        # global_var.__dict__.update(self.all_args_dict)
        # self.global_var = VarClass(**all_args_dict)

        spout_timer_start.click(self.threading_start, inputs=self.all_args_val, outputs=None)
        spout_timer_cancel.click(self.threading_cancel)

        syphon_timer_start.click(self.threading_start, inputs=self.all_args_val, outputs=None)
        syphon_timer_cancel.click(self.threading_cancel)
        # self.spout_image_tx.change(self.send_pick_image, inputs=self.spout_image_tx)
        return self.all_args_val

    # def process(self, p: StableDiffusionProcessingTxt2Img,*args):

    def postprocess(self, p: StableDiffusionProcessingTxt2Img, *args):
        log.warning(f"_____[1][postprocess][enable_spout] p")
        xprint(p.txt2img_image_conditioning)

    # def postprocess_image(self, p, pp, *script_args):
    #     # is_enable=getattr(p,"enable_spout")
    #     log.warning(f"_____[1][postprocess_image][enable_spout] p")
    #     xprint(p)
    #     log.warning(f"_____[2][postprocess_image][enable_spout] pp")
    #     xprint(pp)
    #     # pp.image = p.init_images[0]
    #     # log.warning(f"[2][postprocess_image][enable_spout] {is_enable}")

    # pp.image = ensure_pil_image(pp.image, "RGB")
    # init_image = copy(pp.image)
    # arg_list = self.get_args(p, *args_)
    # params_txt_content = self.read_params_txt()


args_dict = None
args_keys = ['enable_spout', 'enable_spout_tx', 'enable_spout_rx',
             'enable_syphon', 'enable_syphon_tx', 'enable_syphon_rx']
on_image_saved_params = []


def xprint(obj):
    for attr in dir(obj):
        if not attr.startswith("__") and (attr.__contains__('image') or attr.__contains__('img')):
            print(attr + "==>", getattr(obj, attr))


def on_image_saved(params):
    on_image_saved_params.append(params)
    IM_SD_VJ.last_one_image = params
    # call_SpoutGL_sender(on_image_saved_params)
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
