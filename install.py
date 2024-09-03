"""
    .___                        .___           __
  __| _/____   ____ _____     __| _/____     _/  |___  _  __
 / __ |/ __ \_/ ___\\__  \   / __ |/ __ \    \   __\ \/ \/ /
/ /_/ \  ___/\  \___ / __ \_/ /_/ \  ___/     |  |  \     /
\____ |\___  >\___  >____  /\____ |\___  > /\ |__|   \/\_/
     \/    \/     \/     \/      \/    \/  \/
   _____          __           .____    .____       _____
  /  _  \  __ ___/  |_  ____   |    |   |    |     /     \
 /  /_\  \|  |  \   __\/  _ \  |    |   |    |    /  \ /  \
/    |    \  |  /|  | (  <_> ) |    |___|    |___/    Y    \
\____|__  /____/ |__|  \____/  |_______ \_______ \____|__  /
        \/                             \/       \/       \/
             · -—+ auto-prompt-llm-text-vision Extension for ComfyUI +—- ·
             trigger more detail using AI render AI
             https://decade.tw
numpy
SpoutGL
OpenGL_accelerate
PyOpenGL=3.1.7
PyOpenGL_accelerate
"""
import platform

import launch
# if not launch.is_installed("wheel"):
#     launch.run_pip("install wheel", "wheel")
if not launch.is_installed("PyOpenGL"):
    launch.run_pip("install numpy opencv-python PyOpenGL PyOpenGL_accelerate", "install numpy opencv-python PyOpenGL PyOpenGL_accelerate")
if platform.system() == "Windows":
    if not launch.is_installed("SpoutGL"):
        launch.run_pip("install SpoutGL", "SpoutGL")
elif platform.system() == "Darwin":
    if not launch.is_installed("syphon-python"):
        launch.run_pip("install git+https://github.com/cansik/syphon-python/releases/download/v0.1.1/syphon_python-0.1.1-cp310-cp310-macosx_10_9_universal2.whl", "syphon-python")

# if not launch.is_installed("OpenGL_accelerate"):
#     launch.run_pip(f"install -U OpenGL_accelerate", "OpenGL_accelerate")
# if not launch.is_installed("numpy"):
#     launch.run_pip(f"install numpy", "numpy")