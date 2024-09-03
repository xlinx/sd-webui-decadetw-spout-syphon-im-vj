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
import launch

if not launch.is_installed("PyOpenGL"):
    launch.run_pip(f"install SpoutGL PyOpenGL PyOpenGL_accelerate", "SpoutGL PyOpenGL PyOpenGL_accelerate")
# if not launch.is_installed("OpenGL_accelerate"):
#     launch.run_pip(f"install -U OpenGL_accelerate", "OpenGL_accelerate")
# if not launch.is_installed("numpy"):
#     launch.run_pip(f"install numpy", "numpy")