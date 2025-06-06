#!/usr/bin/env python3
"""
Generate sample images using SDXS-512-Dreamshaper model.
Creates sample data for testing the Ray Serve ML inference services.
"""

import os
import sys
from pathlib import Path
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import argparse

def setup_output_directory():
    """Create output directory for sample images."""
    output_dir = Path(__file__).parent.parent / "sample_data"
    output_dir.mkdir(exist_ok=True)
    
    cats_dir = output_dir / "cats_on_desk"
    dogs_dir = output_dir / "dogs_running"
    cats_dir.mkdir(exist_ok=True)
    dogs_dir.mkdir(exist_ok=True)
    
    return output_dir, cats_dir, dogs_dir

def load_model():
    """Load the SDXS-512-Dreamshaper model."""
    print("Loading SDXS-512-Dreamshaper model...")
    
    model_id = "IDKiro/sdxs-512-dreamshaper"
    
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    
    print("Using CPU (model runs fast without CUDA)")
    
    return pipe

def generate_cat_images(pipe, output_dir, num_variations=5):
    """Generate variations of cats sitting on office desks."""
    print(f"Generating {num_variations} cat images...")
    
    variations = [
        "a fluffy orange tabby cat sitting on a modern office desk with a computer",
        "a sleek black cat sitting on a wooden office desk next to a laptop",
        "a gray and white cat sitting on a glass office desk with papers scattered around",
        "a calico cat sitting on a cluttered office desk with books and a coffee mug",
        "a white persian cat sitting elegantly on a clean minimalist office desk"
    ]
    
    generated_files = []
    
    for i, prompt in enumerate(variations[:num_variations], 1):
        print(f"  Generating cat image {i}/{num_variations}: {prompt}")
        
        try:
            image = pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, ugly",
                num_inference_steps=1,
                guidance_scale=0,
                width=512,
                height=512
            ).images[0]
            
            filename = f"cat_on_desk_{i:02d}.png"
            filepath = output_dir / filename
            image.save(filepath)
            generated_files.append(filepath)
            print(f"    Saved: {filepath}")
            
        except Exception as e:
            print(f"    Error generating cat image {i}: {e}")
    
    return generated_files

def generate_dog_images(pipe, output_dir, num_variations=5):
    """Generate variations of dogs running in fields."""
    print(f"Generating {num_variations} dog images...")
    
    variations = [
        "a golden retriever running happily through a green meadow with flowers",
        "a border collie running fast across a grassy field under blue sky",
        "a german shepherd running through a wheat field at sunset",
        "a labrador running in a field of wildflowers with mountains in background",
        "a husky running through a snowy field with pine trees in the distance"
    ]
    
    generated_files = []
    
    for i, prompt in enumerate(variations[:num_variations], 1):
        print(f"  Generating dog image {i}/{num_variations}: {prompt}")
        
        try:
            image = pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, ugly",
                num_inference_steps=1,
                guidance_scale=0,
                width=512,
                height=512
            ).images[0]
            
            filename = f"dog_running_{i:02d}.png"
            filepath = output_dir / filename
            image.save(filepath)
            generated_files.append(filepath)
            print(f"    Saved: {filepath}")
            
        except Exception as e:
            print(f"    Error generating dog image {i}: {e}")
    
    return generated_files

def create_readme(output_dir, cat_files, dog_files):
    """Create a README file describing the sample data."""
    readme_content = """# Sample Data

This directory contains sample images generated using the SDXS-512-Dreamshaper model for testing the Ray Serve ML inference services.


"""
    
    for i, filepath in enumerate(cat_files, 1):
        readme_content += f"- `cats_on_desk/cat_on_desk_{i:02d}.png` - Cat sitting on office desk variation {i}\n"
    
    readme_content += "\n### Dogs Running in Fields\n"
    
    for i, filepath in enumerate(dog_files, 1):
        readme_content += f"- `dogs_running/dog_running_{i:02d}.png` - Dog running in field variation {i}\n"
    
    readme_content += f"""

These images can be used to test the ML inference services:

```bash
curl -X POST -H "Content-Type: application/json" \\
    -d '{{"image_url": "file://sample_data/cats_on_desk/cat_on_desk_01.png",
         "labels": ["a photo of a cat", "a photo of a dog", "a photo of an office"]}}' \\
    http://localhost:8080/

curl -X POST -H "Content-Type: application/json" \\
    -d '{{"image_url": "file://sample_data/dogs_running/dog_running_01.png",
         "labels": ["a photo of a cat", "a photo of a dog", "a photo of a field"]}}' \\
    http://localhost:8080/
```

```bash
curl -X POST -H "Content-Type: application/json" \\
    -d '{{"image_url": "file://sample_data/cats_on_desk/cat_on_desk_01.png",
         "prompt": "What animal is in this image and where is it sitting?"}}' \\
    http://localhost:8081/

curl -X POST -H "Content-Type: application/json" \\
    -d '{{"image_url": "file://sample_data/dogs_running/dog_running_01.png",
         "prompt": "Describe what the animal is doing in this image."}}' \\
    http://localhost:8081/
```

```bash
curl -X POST "http://localhost:8082/text_detect" \\
  -F "image=@sample_data/cats_on_desk/cat_on_desk_01.png" \\
  -F "text=cat desk computer"

curl -X POST "http://localhost:8082/text_detect" \\
  -F "image=@sample_data/dogs_running/dog_running_01.png" \\
  -F "text=dog field grass"
```


Images generated using:
- **Model**: IDKiro/sdxs-512-dreamshaper
- **Resolution**: 512x512 pixels
- **Inference Steps**: 1
- **Guidance Scale**: 7.5

Total images generated: {len(cat_files) + len(dog_files)}
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created README: {readme_path}")
    return readme_path

def main():
    parser = argparse.ArgumentParser(description="Generate sample images for ML inference testing")
    parser.add_argument("--num-variations", type=int, default=5, 
                       help="Number of variations to generate for each category")
    parser.add_argument("--skip-cats", action="store_true", 
                       help="Skip generating cat images")
    parser.add_argument("--skip-dogs", action="store_true", 
                       help="Skip generating dog images")
    
    args = parser.parse_args()
    
    if args.skip_cats and args.skip_dogs:
        print("Error: Cannot skip both cats and dogs!")
        sys.exit(1)
    
    print("🎨 Sample Data Generator for Ray Serve ML Inference")
    print("=" * 50)
    
    output_dir, cats_dir, dogs_dir = setup_output_directory()
    print(f"Output directory: {output_dir}")
    
    try:
        pipe = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install diffusers torch torchvision pillow transformers accelerate")
        sys.exit(1)
    
    cat_files = []
    dog_files = []
    
    if not args.skip_cats:
        try:
            cat_files = generate_cat_images(pipe, cats_dir, args.num_variations)
        except Exception as e:
            print(f"Error generating cat images: {e}")
    
    if not args.skip_dogs:
        try:
            dog_files = generate_dog_images(pipe, dogs_dir, args.num_variations)
        except Exception as e:
            print(f"Error generating dog images: {e}")
    
    readme_path = create_readme(output_dir, cat_files, dog_files)
    
    print("\n✅ Sample data generation complete!")
    print(f"Generated {len(cat_files)} cat images and {len(dog_files)} dog images")
    print(f"Files saved to: {output_dir}")
    print(f"Documentation: {readme_path}")
    
    if cat_files or dog_files:
        print("\n🧪 You can now test the ML services with these sample images!")

if __name__ == "__main__":
    main()
