from base64 import b64encode
import numpy
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login
import gradio as gr

# For video display:
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
import os
import numpy as np  



# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)
token_emb_layer = text_encoder.text_model.embeddings.token_embedding
pos_emb_layer = text_encoder.text_model.embeddings.position_embedding

position_ids = text_encoder.text_model.embeddings.position_ids[:, :77]
position_embeddings = pos_emb_layer(position_ids)


def get_output_embeds(input_embeddings):
    # CLIP's text model uses causal mask, so we prepare it here:
    bsz, seq_len = input_embeddings.shape[:2]
    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, dtype=input_embeddings.dtype)

    # Getting the output embeddings involves calling the model with passing output_hidden_states=True
    # so that it doesn't just return the pooled final predictions:
    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=input_embeddings,
        attention_mask=None, # We aren't using an attention mask so that can be None
        causal_attention_mask=causal_attention_mask.to(torch_device),
        output_attentions=None,
        output_hidden_states=True, # We want the output embs not the final output
        return_dict=None,
    )

    # We're interested in the output hidden state only
    output = encoder_outputs[0]

    # There is a final layer norm we need to pass these through
    output = text_encoder.text_model.final_layer_norm(output)

    # And now they're ready!
    return output


def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32)

def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def generate_with_embs(text_embeddings, text_input, seed,num_inference_steps,guidance_scale):

    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = num_inference_steps # 10            # Number of denoising steps
    guidance_scale = guidance_scale # 7.5                # Scale for classifier-free guidance
    generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
    batch_size = 1

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
      [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    set_timesteps(scheduler, num_inference_steps)

    # Prep latents
    latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # Loop
    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents_to_pil(latents)[0]


def generate_with_prompt_style(prompt, style, seed):

    prompt = prompt + ' in style of s'
    embed = torch.load(style)

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    # for t in text_input['input_ids'][0][:20]: # We'll just look at the first 7 to save you from a wall of '<|endoftext|>'
    #     print(t, tokenizer.decoder.get(int(t)))
    input_ids = text_input.input_ids.to(torch_device)

    token_embeddings = token_emb_layer(input_ids)
    # The new embedding - our special birb word
    replacement_token_embedding = embed[list(embed.keys())[0]].to(torch_device)

    # Insert this into the token embeddings
    token_embeddings[0, torch.where(input_ids[0]==338)] = replacement_token_embedding.to(torch_device)

    # Combine with pos embs
    input_embeddings = token_embeddings + position_embeddings

    #  Feed through to get final output embs
    modified_output_embeddings = get_output_embeds(input_embeddings)

    # And generate an image with this:
    return generate_with_embs(modified_output_embeddings, text_input, seed)

def contrast_loss(images):
    variance = torch.var(images)
    return -variance


def blue_loss(images):
    """
    Computes the blue loss for a batch of images.
    
    The blue loss is defined as the negative variance of the blue channel's pixel values.

    Parameters:
    images (torch.Tensor): A batch of images. Expected shape is (N, C, H, W) where
                           N is the batch size, C is the number of channels (3 for RGB), 
                           H is the height, and W is the width.

    Returns:
    torch.Tensor: The blue loss, which is the negative variance of the blue channel's pixel values.
    """
    # Ensure the input tensor has the correct shape
    if images.shape[1] != 3:
        raise ValueError("Expected images with 3 channels (RGB), but got shape {}".format(images.shape))
    
    # Extract the blue channel (assuming the channels are in RGB order)
    blue_channel = images[:, 2, :, :]
    
    # Calculate the variance of the blue channel
    variance = torch.var(blue_channel)
    
    return -variance


def ymca_loss(images, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Computes the YMCA loss for a batch of images.
    
    The YMCA loss is a custom loss function combining the mean value of the Y (luminance) channel,
    the mean value of the M (magenta) channel, the variance of the C (cyan) channel, and the 
    absolute sum of the A (alpha) channel if present.

    Parameters:
    images (torch.Tensor): A batch of images. Expected shape is (N, C, H, W) where
                           N is the batch size, C is the number of channels (3 for RGB or 4 for RGBA), 
                           H is the height, and W is the width.
    weights (tuple): A tuple of four floats representing the weights for each component of the loss
                     (default is (1.0, 1.0, 1.0, 1.0)).

    Returns:
    torch.Tensor: The YMCA loss, combining the specified components.
    """
    num_channels = images.shape[1]
    
    if num_channels not in [3, 4]:
        raise ValueError("Expected images with 3 (RGB) or 4 (RGBA) channels, but got shape {}".format(images.shape))
    
    # Extract the RGB channels
    R = images[:, 0, :, :]
    G = images[:, 1, :, :]
    B = images[:, 2, :, :]
    
    # Convert RGB to Y (luminance) channel
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    
    # Convert RGB to M (magenta) channel
    M = 1 - G
    
    # Convert RGB to C (cyan) channel
    C = 1 - R
    
    # Compute the mean of the Y channel
    mean_Y = torch.mean(Y)
    
    # Compute the mean of the M channel
    mean_M = torch.mean(M)
    
    # Compute the variance of the C channel
    variance_C = torch.var(C)
    
    loss = weights[0] * mean_Y + weights[1] * mean_M - weights[2] * variance_C
    
    if num_channels == 4:
        # Extract the alpha channel
        A = images[:, 3, :, :]
        # Compute the absolute sum of the A channel
        abs_sum_A = torch.sum(torch.abs(A))
        # Include the alpha component in the loss
        loss += weights[3] * abs_sum_A
    
    return loss



def rgb_to_cmyk(images):
    """
    Converts an RGB image tensor to CMYK.

    Parameters:
    images (torch.Tensor): A batch of images in RGB format. Expected shape is (N, 3, H, W).

    Returns:
    torch.Tensor: A tensor containing the CMYK channels.
    """
    R = images[:, 0, :, :]
    G = images[:, 1, :, :]
    B = images[:, 2, :, :]

    # Convert RGB to CMY
    C = 1 - R
    M = 1 - G
    Y = 1 - B

    # Convert CMY to CMYK
    K = torch.min(torch.min(C, M), Y)
    C = (C - K) / (1 - K + 1e-8)
    M = (M - K) / (1 - K + 1e-8)
    Y = (Y - K) / (1 - K + 1e-8)

    CMYK = torch.stack([C, M, Y, K], dim=1)
    return CMYK

def cymk_loss(images, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Computes the CYMK loss for a batch of images.
    
    The CYMK loss is a custom loss function combining the variance of the Cyan channel,
    the mean value of the Yellow channel, the variance of the Magenta channel, and the 
    absolute sum of the Black channel.

    Parameters:
    images (torch.Tensor): A batch of images. Expected shape is (N, 3, H, W) for RGB input.
    weights (tuple): A tuple of four floats representing the weights for each component of the loss
                     (default is (1.0, 1.0, 1.0, 1.0)).

    Returns:
    torch.Tensor: The CYMK loss, combining the specified components.
    """
    # Ensure the input tensor has the correct shape
    if images.shape[1] != 3:
        raise ValueError("Expected images with 3 channels (RGB), but got shape {}".format(images.shape))
    
    # Convert RGB to CMYK
    cmyk_images = rgb_to_cmyk(images)
    
    # Extract CMYK channels
    C = cmyk_images[:, 0, :, :]
    M = cmyk_images[:, 1, :, :]
    Y = cmyk_images[:, 2, :, :]
    K = cmyk_images[:, 3, :, :]
    
    # Compute the variance of the C channel
    variance_C = torch.var(C)
    
    # Compute the mean of the Y channel
    mean_Y = torch.mean(Y)
    
    # Compute the variance of the M channel
    variance_M = torch.var(M)
    
    # Compute the absolute sum of the K channel
    abs_sum_K = torch.sum(torch.abs(K))
    
    # Combine the components with the given weights
    loss = (weights[0] * variance_C) + (weights[1] * mean_Y) + (weights[2] * variance_M) + (weights[3] * abs_sum_K)
    
    return loss


def blue_loss_variant(images, use_mean=False, alpha=1.0):
    """
    Computes the blue loss for a batch of images with an optional mean component.
    
    The blue loss is defined as the negative variance of the blue channel's pixel values.
    Optionally, it can also include the mean value of the blue channel.

    Parameters:
    images (torch.Tensor): A batch of images. Expected shape is (N, C, H, W) where
                           N is the batch size, C is the number of channels (3 for RGB), 
                           H is the height, and W is the width.
    use_mean (bool): If True, includes the mean of the blue channel in the loss calculation.
    alpha (float): Weighting factor for the mean component when use_mean is True.

    Returns:
    torch.Tensor: The blue loss, which is the negative variance of the blue channel's pixel values,
                  optionally combined with the mean value of the blue channel.
    """
    # Ensure the input tensor has the correct shape
    if images.shape[1] != 3:
        raise ValueError("Expected images with 3 channels (RGB), but got shape {}".format(images.shape))
    
    # Extract the blue channel (assuming the channels are in RGB order)
    blue_channel = images[:, 2, :, :]
    
    # Calculate the variance of the blue channel
    variance = torch.var(blue_channel)
    
    if use_mean:
        # Calculate the mean of the blue channel
        mean = torch.mean(blue_channel)
        # Combine variance and mean into the loss
        loss = -variance + alpha * mean
    else:
        loss = -variance
    
    return loss

def generate_with_prompt_style_guidance(prompt, style, seed,num_inference_steps,guidance_scale,loss_function):

    prompt = prompt + ' in style of s'
    
    embed = torch.load(style)

    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = num_inference_steps  #           # Number of denoising steps
    guidance_scale = guidance_scale #               # Scale for classifier-free guidance
    generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
    batch_size = 1
    

    # Prep text
    text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    input_ids = text_input.input_ids.to(torch_device)

    # Get token embeddings
    token_embeddings = token_emb_layer(input_ids)

    # The new embedding - our special birb word
    replacement_token_embedding = embed[list(embed.keys())[0]].to(torch_device)

    # Insert this into the token embeddings
    token_embeddings[0, torch.where(input_ids[0]==338)] = replacement_token_embedding.to(torch_device)

    # Combine with pos embs
    input_embeddings = token_embeddings + position_embeddings

    #  Feed through to get final output embs
    modified_output_embeddings = get_output_embeds(input_embeddings)

    # And the uncond. input as before:
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, modified_output_embeddings])

    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # Loop
    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        #### ADDITIONAL GUIDANCE ###
        if i%5 == 0:
            # Requires grad on the latents
            latents = latents.detach().requires_grad_()

            # Get the predicted x0:
            latents_x0 = latents - sigma * noise_pred
            # latents_x0 = scheduler.step(noise_pred, t, latents).pred_original_sample

            # Decode to image space
            denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 # range (0, 1)

            # Calculate loss
            # "contrast", "blue_original", "blue_modified","ymca_loss","cymk_loss"
            if loss_function == "contrast":
                loss_scale = 200 #
                loss = contrast_loss(denoised_images) * loss_scale
            elif loss_function == "blue_original":
                loss_scale = 200 #
                loss = blue_loss(denoised_images) * loss_scale
            elif loss_function == "blue_modified":
                loss_scale = 200 #
                loss = blue_loss_variant(denoised_images) * loss_scale
            elif loss_function == "ymca":
                loss_scale = 200 #
                loss = ymca_loss(denoised_images) * loss_scale
            elif loss_function == "cmyk":
                loss_scale = 1 #
                loss = cymk_loss(denoised_images) * loss_scale
            else :
                loss_scale = 200
                loss = ymca_loss(denoised_images) * loss_scale

            # # Occasionally print it out
            # if i%10==0:
            #     print(i, 'loss:', loss.item())

            # Get gradient
            cond_grad = torch.autograd.grad(loss, latents)[0]

            # Modify the latents based on this gradient
            latents = latents.detach() - cond_grad * sigma**2

        # Now step with scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample


    return latents_to_pil(latents)[0]




dict_styles = { 
    'Dr Strange': 'styles/learned_embeds_dr_strange.bin', 
    'GTA-5':'styles/learned_embeds_gta5.bin', 
    'Manga':'styles/learned_embeds_manga.bin', 
    'Pokemon':'styles/learned_embeds_pokemon.bin',
    'Illustration': 'styles/learned_embeds_illustration.bin', 
    'Matrix':'styles/learned_embeds_matrix.bin', 
    'Oil Painting':'styles/learned_embeds_oil.bin',  
}

def inference(prompt, seed, style,num_inference_steps,guidance_scale,loss_function): 
    
    if prompt is not None and style is not None and seed is not None: 
        print(loss_function)
        style = dict_styles[style]
        torch.manual_seed(seed)
        result = generate_with_prompt_style_guidance(prompt, style,seed,num_inference_steps,guidance_scale,loss_function)
        return np.array(result)
    else:
        return None

title = "Stable Diffusion and Textual Inversion"
description = "Gradio interface to apply style to Stable Diffusion outputs"
examples = [["Pink Ferrari Car", 24041975,"Manga"], ["A man sipping tea wearing a spacesuit on the moon",24041975, "GTA-5"]]  # Added valid styles

demo = gr.Interface(inference,
                    inputs = [gr.Textbox(label='Prompt', value='Pink Ferrari Car'),    gr.Textbox(label='Seed', value=24041975),                           
                              gr.Dropdown(['Dr Strange', 'GTA-5', 'Manga', 'Pokemon','Illustration','Matrix','Oil Painting'], label='Style', value='Dr Strange'),
                              gr.Slider(
                                minimum=5,
                                maximum=20,
                                value=10,
                                step=5,
                                label="Select Number of Steps",
                                interactive=True,
                                ),
                                gr.Slider(
                                minimum=0,
                                maximum=10,
                                value=8,
                                step=8,
                                label="Select Guidance Scale",
                                interactive=True,
                                ),gr.Radio(["contrast", "blue_original", "blue_modified","ymca","cmyk"], label="loss-function", info="loss-function" , value="ymca"),
                              ],
                    outputs = [
                              gr.Image(label="Stable Diffusion Output"),
                              ],
                    title = title,
                    description = description,
                    # examples = examples, 
                    # cache_examples=True
                    )
demo.launch()
     
