from diffusers import DiffusionPipeline, AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained("/home/mvchaychuk/kandinsky-2-2-decoder", local_files_only=True)
pipeline.to('cuda')
image = pipeline("gummy bear", num_inference_steps=30).images[0]