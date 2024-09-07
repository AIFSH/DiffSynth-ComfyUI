# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./js"

from .util_nodes import LoadVideo,PreViewVideo
from .video_synthesis_nodes import DownloadModelsNode, CogVideoNode,TextEncode

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadVideo": LoadVideo,
    "PreViewVideo":PreViewVideo,
    "DownloadModelsNode":DownloadModelsNode,
    "CogVideoNode":CogVideoNode,
    "TextEncode":TextEncode
}
