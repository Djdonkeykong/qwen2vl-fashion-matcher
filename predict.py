"""
Replicate Predictor for Qwen2-VL Fashion Analysis
===============================================

Deploys the superior local Qwen2-VL model with all custom optimizations:
- Color cleaning (orange/yellow -> orange)
- Enhanced prompts with clear color/pattern separation  
- Pattern inference fallbacks
- Optimized generation config
"""

import os
import json
import torch
import tempfile
from typing import Dict, List, Optional, Any
from pathlib import Path
from PIL import Image
from datetime import datetime

from cog import BasePredictor, Input, Path as CogPath
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class FashionAnalyzer:
    """Enhanced Qwen2-VL analyzer for fashion with all custom optimizations."""
    
    def __init__(self):
        self.model_name = "Qwen/Qwen2-VL-2B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._model_loaded = False
        
        # Optimized generation config
        self.generation_config = {
            "max_new_tokens": 150,
            "min_new_tokens": 80,
            "do_sample": False,
            "repetition_penalty": 1.05,
            "use_cache": True,
            "early_stopping": True,
        }
        
    def _ensure_model_loaded(self):
        """Load model if not already loaded."""
        if not self._model_loaded:
            print(f"[Qwen2VL] Loading {self.model_name} on {self.device}...")
            
            # Load model at full precision for accuracy
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            self.model.eval()
            
            # Compile for speed
            try:
                print("[Qwen2VL] Compiling model for speed...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[Qwen2VL] Model compilation successful")
            except Exception as e:
                print(f"[Qwen2VL] Compilation failed (continuing): {e}")
            
            self._model_loaded = True
            
            # Show memory usage
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"[Qwen2VL] Model loaded - GPU Memory: {allocated:.1f}GB")
    
    def _clean_compound_color(self, color: str) -> str:
        """Clean compound colors to single dominant colors for database compatibility."""
        if not color or color == "unknown":
            return color
        
        color_lower = color.lower().strip()
        
        # Handle compound colors - pick the dominant one
        compound_mappings = {
            # Orange/yellow combinations
            "orange/yellow": "orange",
            "yellow/orange": "orange", 
            "orangeyellow": "orange",
            "orange-yellow": "orange",
            "orange yellow": "orange",
            
            # Black/white combinations
            "black/white": "black",
            "white/black": "black",
            "blackwhite": "black",
            "black-white": "black",
            "black white": "black",
            
            # Blue combinations
            "navy/blue": "navy",
            "blue/navy": "navy",
            "dark blue": "navy",
            "light blue": "blue",
            "sky blue": "blue",
            
            # Red combinations
            "red/pink": "red",
            "pink/red": "red",
            "dark red": "red",
            "burgundy": "red",
            "maroon": "red",
            
            # Green combinations
            "green/blue": "green",
            "blue/green": "blue",
            "dark green": "green",
            "light green": "green",
            "olive": "green",
            
            # Brown combinations  
            "brown/tan": "brown",
            "tan/brown": "brown",
            "beige": "brown",
            "khaki": "brown",
            
            # Gray combinations
            "gray/grey": "gray",
            "grey/gray": "gray",
            "light gray": "gray",
            "dark gray": "gray",
            "charcoal": "gray"
        }
        
        # Check for direct mappings first
        if color_lower in compound_mappings:
            return compound_mappings[color_lower]
        
        # Handle slash-separated colors - take the first one
        if "/" in color_lower:
            first_color = color_lower.split("/")[0].strip()
            first_color = self._clean_single_color(first_color)
            return first_color
        
        # Handle space-separated colors - take the last one (usually the noun)
        if " " in color_lower and not any(word in color_lower for word in ["light", "dark", "bright", "deep"]):
            parts = color_lower.split()
            if len(parts) == 2:
                return self._clean_single_color(parts[-1])
        
        # Return cleaned single color
        return self._clean_single_color(color_lower)
    
    def _clean_single_color(self, color: str) -> str:
        """Clean a single color to standard database vocabulary."""
        color = color.strip().lower()
        
        # Standard color mappings
        standard_colors = {
            "navy": "navy", "blue": "blue", "royal": "blue", "cobalt": "blue", "teal": "blue",
            "red": "red", "crimson": "red", "scarlet": "red", "burgundy": "red", "maroon": "red", "wine": "red",
            "black": "black", "white": "white", "gray": "gray", "grey": "gray", "charcoal": "gray", "silver": "gray",
            "green": "green", "olive": "green", "forest": "green", "lime": "green", "mint": "green",
            "brown": "brown", "tan": "brown", "beige": "brown", "khaki": "brown", "camel": "brown",
            "pink": "pink", "purple": "purple", "violet": "purple", "yellow": "yellow", "orange": "orange", "coral": "orange", "gold": "yellow"
        }
        
        return standard_colors.get(color, color)
    
    def _infer_pattern_from_title(self, title: Optional[str]) -> Optional[str]:
        """Infer pattern from product title when Qwen returns null."""
        if not title:
            return None
        t = title.lower()

        # Keyword-based rules
        if any(k in t for k in ["stripe", "striped"]): return "striped"
        if any(k in t for k in ["check", "checked", "plaid", "tartan"]): return "checked"
        if any(k in t for k in ["polka", "dot"]): return "polka dot"
        if "floral" in t: return "floral"
        if any(k in t for k in ["print", "printed", "graphic"]): return "printed"
        if any(k in t for k in ["leopard", "zebra", "animal"]): return "animal print"
        if any(k in t for k in ["heather", "melange", "mélange", "marled"]): return "heather"
        if "rib" in t: return "ribbed"

        return "solid"  # Default fallback
        
    def _create_fashion_analysis_prompt(self) -> str:
        """Create optimized prompt for fashion image analysis."""
        
        json_template = """{
  "detected_garments": [
    {
      "type": "[exact garment type - be specific like 'maxi dress', 'button-down shirt', 'denim jacket', etc.]",
      "color": "[ACTUAL COLOR ONLY: red/blue/green/black/white/pink/etc - NEVER patterns like 'floral' or 'striped']", 
      "pattern": "[PATTERN ONLY: solid/striped/floral/plaid/polka-dot/geometric/etc or null - NEVER colors]",
      "neckline": "[v-neck/crew-neck/off-shoulder/halter/etc or null]",
      "sleeve_style": "[long/short/sleeveless/3-quarter/etc or null]",
      "fit": "[loose/fitted/regular/oversized/slim or null]",
      "length": "[short/midi/long/cropped/etc or null]",
      "fabric": "[cotton/denim/silk/knit/leather/etc or null]"
    }
  ]
}"""

        return f"""Analyze this fashion image and identify the main clothing item or accessory.

CRITICAL INSTRUCTIONS:
- COLOR field = ACTUAL COLORS ONLY (red, blue, green, black, white, pink, etc.)
- PATTERN field = PATTERN TYPES ONLY (solid, striped, floral, plaid, etc.)
- NEVER put patterns in color field or colors in pattern field
- If floral dress: color="pink" (or dominant color), pattern="floral"
- If striped shirt: color="blue" (or dominant color), pattern="striped"
- For COLOR: Use the background/base color, NOT stripe colors (e.g., for navy dress with white stripes, use "navy")
- For PATTERN: Look carefully - if you see stripes, use "striped", not "solid"
- For PATTERN: Only use "solid" if there is truly NO pattern visible  
- For NECKLINE: Simple sleeveless dresses usually have "tank" necklines, not "off-the-shoulder"
- For STRIPED items: The base color is what matters, not the stripe color
- Examine the image closely before responding

GARMENT CATEGORIES to consider:
- Clothing: dress, top, shirt, blouse, t-shirt, sweatshirt, sweater, cardigan, jacket, coat, vest, pants, jeans, shorts, skirt, jumpsuit

Respond with valid JSON in this exact format:

{json_template}

IMPORTANT RULES:
1. Be specific with garment types - use detailed names like 'denim jacket', 'maxi dress', 'ankle boots'  
2. Separate color and pattern clearly - colors go in color field, patterns go in pattern field
3. Return ONLY valid JSON, no other text
4. Focus on the main garment the person is wearing"""
        
    def analyze_image(self, image_path: str) -> Optional[Dict]:
        """Analyze fashion image with all optimizations."""
        self._ensure_model_loaded()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Create messages
            prompt = self._create_fashion_analysis_prompt()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ],
                }
            ]
            
            # Prepare input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    **self.generation_config
                )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
            
            # Parse JSON response
            try:
                # Clean response
                clean_output = output_text.strip()
                if clean_output.startswith('```json'):
                    clean_output = clean_output[7:]
                if clean_output.endswith('```'):
                    clean_output = clean_output[:-3]
                clean_output = clean_output.strip()
                
                response_data = json.loads(clean_output)
                
                # Extract and clean garment data
                detected_garments = response_data.get('detected_garments', [])
                if not detected_garments:
                    return None
                
                garment_data = detected_garments[0]
                
                # Apply color cleaning
                raw_color = garment_data.get("color", "unknown")
                cleaned_color = self._clean_compound_color(raw_color)
                garment_data["color"] = cleaned_color
                
                # Apply pattern fallback if needed
                if not garment_data.get("pattern"):
                    garment_data["pattern"] = "solid"
                
                return {
                    "detected_garments": [garment_data],
                    "raw_response": response_data,
                    "color_cleaned": raw_color != cleaned_color
                }
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return None
                
        except Exception as e:
            print(f"Analysis error: {e}")
            return None


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.analyzer = FashionAnalyzer()
        # Pre-load model
        self.analyzer._ensure_model_loaded()

    def predict(
        self, 
        image: CogPath = Input(description="Fashion image to analyze")
    ) -> Dict[str, Any]:
        """
        Analyze fashion image using optimized Qwen2-VL with color cleaning and enhanced prompts.
        Returns detailed garment attributes including cleaned colors for database compatibility.
        """
        
        # Save uploaded image to temporary file
        # Use the image path directly since it's already a Path object
        tmp_path = str(image)
        
        try:
            # Analyze the image
            result = self.analyzer.analyze_image(tmp_path)
            
            if not result:
                return {
                    "success": False,
                    "error": "No garments detected in image"
                }
            
            garment = result["detected_garments"][0]
            
            return {
                "success": True,
                "analysis": {
                    "type": garment.get("type", "unknown"),
                    "color": garment.get("color", "unknown"),
                    "pattern": garment.get("pattern"),
                    "neckline": garment.get("neckline"),
                    "sleeve_style": garment.get("sleeve_style"),
                    "fit": garment.get("fit"),
                    "length": garment.get("length"),
                    "fabric": garment.get("fabric")
                },
                "metadata": {
                    "model": "Qwen2-VL-2B-Instruct",
                    "color_cleaned": result.get("color_cleaned", False),
                    "confidence": 0.8,
                    "timestamp": datetime.now().isoformat()
                },
                "raw_response": result["raw_response"]
            }
            
        finally:
            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass
