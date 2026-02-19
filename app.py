import os
import io
import base64
import time
import hydra
from omegaconf import OmegaConf
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
import numpy as np
import logging

from scripts.model_config import model_selection

# Configuration
CONFIG_PATH = "conf"
CONFIG_NAME = "config_supresdiffgan_evaluation.yaml"
INFERENCE_STEPS = 100
INFERENCE_POSTERIOR = "ddim"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global Variables
app = Flask(__name__, static_folder='static')
CORS(app)
model = None
device = None
lr_transform = None
cfg = None

def load_model_and_config():
    """Loads the Hydra config and the PyTorch model."""
    global model, device, lr_transform, cfg
    
    try:
        logger.info("Initializing Hydra configuration...")
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path=CONFIG_PATH, version_base=None)
        cfg = hydra.compose(config_name=CONFIG_NAME)
        
        logger.info(f"Configuration loaded from {CONFIG_PATH}/{CONFIG_NAME}")
        
        if cfg.model.load_model is None:
            raise ValueError("`model.load_model` path is not specified in the config.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Set precision
        precision = cfg.trainer.get('precision', '32-true')
        if precision in ['16-mixed', 'bf16-mixed']:
            torch.set_float32_matmul_precision('medium')
            logger.info(f"Using mixed precision: {precision}")
        
        # Initialize model
        logger.info("Initializing model architecture...")
        model_instance = model_selection(cfg=cfg, device=device)
        
        # Load checkpoint
        checkpoint_path = cfg.model.load_model
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model_instance.load_state_dict(state_dict, strict=False)
        else:
            model_instance.load_state_dict(checkpoint, strict=False)
        
        model = model_instance
        model.to(device)
        model.eval()
        
        # Configure diffusion
        model.diffusion.set_timesteps(INFERENCE_STEPS)
        model.diffusion.set_posterior_type(INFERENCE_POSTERIOR)
        logger.info(f"Diffusion settings - Steps: {INFERENCE_STEPS}, Posterior: {INFERENCE_POSTERIOR}")
        
        # Define transformations
        lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        logger.info("✓ Model loaded successfully and ready for inference")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise

def pad_to_multiple(image, multiple=64):
    """Pad image to make dimensions divisible by multiple.
    
    Using multiple=64 to ensure compatibility with deep UNet architectures
    that have multiple downsampling/upsampling layers (typically 4-5 levels,
    requiring divisibility by 2^4=16 or 2^5=32, but 64 is safer).
    """
    width, height = image.size
    
    # Calculate padding needed
    new_width = ((width + multiple - 1) // multiple) * multiple
    new_height = ((height + multiple - 1) // multiple) * multiple
    
    # Calculate padding amounts
    pad_width = new_width - width
    pad_height = new_height - height
    
    # Pad symmetrically (or mostly symmetrically)
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    
    # Create padded image - use edge replication for better quality
    padded = Image.new('RGB', (new_width, new_height), (128, 128, 128))
    padded.paste(image, (left, top))
    
    # Fill padding areas by replicating edge pixels
    # Left edge
    if left > 0:
        edge_left = image.crop((0, 0, 1, height))
        for i in range(left):
            padded.paste(edge_left, (i, top))
    
    # Right edge
    if right > 0:
        edge_right = image.crop((width-1, 0, width, height))
        for i in range(right):
            padded.paste(edge_right, (left + width + i, top))
    
    # Top edge
    if top > 0:
        edge_top = padded.crop((0, top, new_width, top+1))
        for i in range(top):
            padded.paste(edge_top, (0, i))
    
    # Bottom edge
    if bottom > 0:
        edge_bottom = padded.crop((0, top + height - 1, new_width, top + height))
        for i in range(bottom):
            padded.paste(edge_bottom, (0, top + height + i))
    
    return padded, (left, top, right, bottom)

def resize_to_compatible_size(image, base_size=256):
    """Resize image to a size that's known to work with the model.
    
    The model was likely trained on specific sizes (e.g., 64x64 LR -> 256x256 HR).
    This function resizes to a compatible size while preserving aspect ratio.
    """
    width, height = image.size
    aspect_ratio = width / height
    
    # Determine new dimensions based on aspect ratio
    if aspect_ratio > 1:
        # Landscape
        new_width = base_size
        new_height = int(base_size / aspect_ratio)
    else:
        # Portrait or square
        new_height = base_size
        new_width = int(base_size * aspect_ratio)
    
    # Round to nearest multiple of 64
    new_width = ((new_width + 63) // 64) * 64
    new_height = ((new_height + 63) // 64) * 64
    
    # Resize image
    resized = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized, (new_width, new_height)

def preprocess_image(image_bytes):
    """Preprocesses image bytes for the model."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image.size
        
        # Try two approaches:
        # 1. First try padding to multiple of 64
        padded_image, padding = pad_to_multiple(image, multiple=64)
        
        # Check if dimensions are reasonable (not too large)
        max_dim = 1024  # Maximum dimension for latent space
        if padded_image.size[0] > max_dim or padded_image.size[1] > max_dim:
            logger.warning(f"Image too large after padding: {padded_image.size}. Resizing instead.")
            # 2. If too large, resize to compatible size
            processed_image, processed_size = resize_to_compatible_size(image, base_size=256)
            padding = (0, 0, 0, 0)  # No padding used
        else:
            processed_image = padded_image
            processed_size = padded_image.size
        
        logger.info(f"Preprocessing: Original {original_size} -> Processed {processed_size}, Padding: {padding}")
        
        tensor = lr_transform(processed_image).unsqueeze(0)
        return tensor, image, original_size, padding, processed_size
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        return None, None, None, None, None

def run_inference(lr_tensor):
    """Runs super-resolution inference."""
    if model is None or device is None:
        logger.error("Model not loaded")
        return None
    
    try:
        logger.info(f"Input tensor shape: {lr_tensor.shape}")
        
        with torch.no_grad():
            lr_tensor = lr_tensor.to(device)
            precision = cfg.trainer.get('precision', '32-true')
            
            if precision in ['16-mixed', 'bf16-mixed']:
                with torch.amp.autocast('cuda'):
                    sr_tensor = model(lr_tensor)
            else:
                sr_tensor = model(lr_tensor)
            
            logger.info(f"Output tensor shape: {sr_tensor.shape}")
            return sr_tensor.cpu()
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        logger.error(f"Input tensor shape was: {lr_tensor.shape if lr_tensor is not None else 'None'}")
        return None

def postprocess_image(tensor, padding, original_size, scale_factor=4):
    """Converts output tensor to PIL image and handles cropping/resizing."""
    if tensor is None:
        return None
    
    try:
        img_np = tensor.squeeze(0).mul(0.5).add(0.5).clamp(0, 1).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        image = Image.fromarray(img_np)
        
        # If padding was used, remove it
        if padding != (0, 0, 0, 0):
            left, top, right, bottom = padding
            left_scaled = left * scale_factor
            top_scaled = top * scale_factor
            right_scaled = right * scale_factor
            bottom_scaled = bottom * scale_factor
            
            width, height = image.size
            crop_box = (
                left_scaled,
                top_scaled,
                width - right_scaled,
                height - bottom_scaled
            )
            
            image = image.crop(crop_box)
            logger.info(f"Cropped from {width}x{height} to {image.size}")
        
        # Optionally resize to expected output dimensions based on original
        expected_width = original_size[0] * scale_factor
        expected_height = original_size[1] * scale_factor
        
        # Only resize if dimensions don't match expected (with some tolerance)
        if abs(image.size[0] - expected_width) > 5 or abs(image.size[1] - expected_height) > 5:
            logger.info(f"Resizing output from {image.size} to ({expected_width}, {expected_height})")
            image = image.resize((expected_width, expected_height), Image.LANCZOS)
        
        return image
    except Exception as e:
        logger.error(f"Error during postprocessing: {e}", exc_info=True)
        return None

def image_to_base64(image):
    """Converts PIL image to base64 data URL."""
    if image is None:
        return None
    
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error encoding to base64: {e}", exc_info=True)
        return None

# Flask Routes
@app.route('/')
def index():
    """Serve the main HTML page."""
    try:
        return send_from_directory('.', 'super_resolution_ui.html')
    except Exception as e:
        logger.error(f"Error serving index: {e}", exc_info=True)
        return "Error: UI file not found.", 404

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    })

@app.route('/model-info')
def model_info():
    """Model information endpoint."""
    if model is None or cfg is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    info = {
        "model_name": cfg.model.name,
        "scale_factor": cfg.dataset.get('scale', 4),
        "diffusion_steps": INFERENCE_STEPS,
        "diffusion_posterior": INFERENCE_POSTERIOR,
        "device": str(device),
        "precision": cfg.trainer.get('precision', '32-true'),
        "checkpoint": cfg.model.load_model
    }
    
    return jsonify(info)

@app.route('/upscale', methods=['POST'])
def upscale_image():
    """Upscale endpoint for image enhancement."""
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file in request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        img_bytes = file.read()
        logger.info(f"Processing: {file.filename} ({len(img_bytes)} bytes)")
        
        # Preprocess
        result = preprocess_image(img_bytes)
        if result[0] is None:
            return jsonify({"error": "Failed to preprocess image"}), 500
        
        lr_tensor, original_pil, original_size, padding, padded_size = result
        logger.info(f"Original size: {original_size}, Padded size: {padded_size}, Padding: {padding}")
        
        # Inference
        logger.info("Running inference...")
        start_time = time.time()
        sr_tensor = run_inference(lr_tensor)
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f}s")
        
        if sr_tensor is None:
            return jsonify({"error": "Inference failed"}), 500
        
        # Postprocess (remove padding and scale)
        scale_factor = cfg.dataset.get('scale', 4)
        sr_image_pil = postprocess_image(sr_tensor, padding, original_size, scale_factor)
        if sr_image_pil is None:
            return jsonify({"error": "Failed to postprocess"}), 500
        
        logger.info(f"SR image size after crop: {sr_image_pil.size}")
        
        # Encode images
        original_base64 = image_to_base64(original_pil)
        sr_base64 = image_to_base64(sr_image_pil)
        
        if not original_base64 or not sr_base64:
            return jsonify({"error": "Failed to encode images"}), 500
        
        logger.info("✓ Image processed successfully")
        
        return jsonify({
            "original_image": original_base64,
            "sr_image": sr_base64,
            "original_size": original_size,
            "sr_size": sr_image_pil.size,
            "inference_time": round(inference_time, 2),
            "scale_factor": scale_factor
        })
    
    except Exception as e:
        logger.error(f"Error during upscaling: {e}", exc_info=True)
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

if __name__ == '__main__':
    try:
        logger.info("=" * 60)
        logger.info("Starting Vision Weaver Super-Resolution Server")
        logger.info("=" * 60)
        load_model_and_config()
        logger.info("Server starting on http://127.0.0.1:5000")
        app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
    except Exception as e:
        logger.critical(f"Failed to start: {e}", exc_info=True)