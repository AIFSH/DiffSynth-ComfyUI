import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)

import time
import torch
import folder_paths
from diffsynth.extensions.RIFE import RIFEInterpolater
from diffsynth import download_models,ModelManager\
    ,CogVideoPipeline,save_video,VideoData

output_dir = folder_paths.get_output_directory()

class DownloadModelsNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model_id_list":("STRING",{
                    "default":"CogVideoX-5B,RIFE",
                    "multiline":True
                }),
            }
        }
    
    RETURN_TYPES = ("FILES",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "download"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_DiffSynth"

    def download(self,model_id_list):
        model_id_list = model_id_list.split(",")
        return (download_models(model_id_list=model_id_list),)


class TextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
            }
        }
    RETURN_TYPES = ("TEXT",)
    OUTPUT_TOOLTIPS = ("the text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "AIFSH_DiffSynth"
    DESCRIPTION = "return a text prompt that can be used to guide the diffusion model towards generating specific images."

    def encode(self,text):
        return (text, )

class CogVideoNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt":("TEXT",),
                "if_upscale":("BOOLEAN",{
                    "default": False,
                }),
                "if_interpolate":("BOOLEAN",{
                    "default": False,
                }),
                "seed":("INT",{
                    "default":42,
                })
            },
            "optional":{
                "video":("VIDEO",),
                "models":("FILES",)
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_DiffSynth"

    def gen_video(self,prompt,if_upscale,if_interpolate,seed,video=None,models=None):
        model_manager = ModelManager(torch_dtype=torch.bfloat16)
        model_manager.load_models([
            "models/CogVideo/CogVideoX-5b/text_encoder",
            "models/CogVideo/CogVideoX-5b/transformer",
            "models/CogVideo/CogVideoX-5b/vae/diffusion_pytorch_model.safetensors",
            "models/RIFE/flownet.pkl",
        ])

        pipe = CogVideoPipeline.from_model_manager(model_manager)
        torch.manual_seed(seed)

        input_video = None
        denoising_strength = 1.0
        width=480
        height = 720
        num_inference_steps=200
        tiled = False
        if video:
            input_video = VideoData(video_file=video)
            denoising_strength = 0.7

            if if_upscale:
                height=480*2
                width = 720*2
                denoising_strength = 0.4
                num_inference_steps = 30
                tiled = True
            

        images = pipe(prompt=prompt,
                      width=width,
                      height=height,
                      num_inference_steps=num_inference_steps,
                      input_video=input_video,
                      denoising_strength=denoising_strength,
                      tiled=tiled)
        
        video_path = os.path.join(output_dir,f"diffsynth_cogvideo_{time.time_ns()}.mp4")
        save_video(images,video_path,fps=8,quality=5)

        if if_interpolate:
            rife = RIFEInterpolater.from_model_manager(model_manager)
            video = VideoData(video_file=video_path).raw_data()
            video = rife.interpolate(video,num_iter=2)
            video_path = video_path.split(".")[0] + "rife.mp4"
            save_video(video,video_path,fps=32,quality=5)

        return (video_path,)



