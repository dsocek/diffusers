import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("../../../FLUX.1-dev", torch_dtype=torch.bfloat16)
#pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe = pipe.to("cuda")

# Uncomment to use torch.compile() mode
#pipe = torch.compile(pipe)

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=30,
    max_sequence_length=512,
).images[0]
image.save("flux-dev.png")
