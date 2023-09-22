from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from diffusers import DiffusionPipeline  # Substitua pelo import correto em seu ambiente
import torch
from PIL import Image
import io
import base64

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Verifique se CUDA está disponível, senão use CPU
# Verifique se CUDA está disponível, senão use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Especifica o dtype com base no dispositivo
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Inicialize os modelos DiffusionPipeline
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype, variant="fp16" if device == "cuda" else "fp32", use_safetensors=True
)
base.to(device)  # Mova para GPU se disponível, senão mantenha em CPU

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch_dtype,
    use_safetensors=True,
    variant="fp16" if device == "cuda" else "fp32",
)
refiner.to(device) 
class PromptInput(BaseModel):
    prompt: str

class GeneratedImage(BaseModel):
    image: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_model=GeneratedImage)
async def generate_image(prompt_input: PromptInput):
    prompt = prompt_input.prompt
    n_steps = 60
    high_noise_frac = 0.86
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    # Converter a imagem em base64 para envio
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format="PNG")
    image_base64 = base64.b64encode(image_byte_array.getvalue()).decode()

    return {"image": image_base64}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888, reload=True, workers=1)
