# Sample Data

This directory contains sample images generated using the SDXS-512-Dreamshaper model for testing the Ray Serve ML inference services.



### Dogs Running in Fields
- `dogs_running/dog_running_01.png` - Dog running in field variation 1


These images can be used to test the ML inference services:

```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"image_url": "file://sample_data/cats_on_desk/cat_on_desk_01.png",
         "labels": ["a photo of a cat", "a photo of a dog", "a photo of an office"]}' \
    http://localhost:8080/

curl -X POST -H "Content-Type: application/json" \
    -d '{"image_url": "file://sample_data/dogs_running/dog_running_01.png",
         "labels": ["a photo of a cat", "a photo of a dog", "a photo of a field"]}' \
    http://localhost:8080/
```

```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"image_url": "file://sample_data/cats_on_desk/cat_on_desk_01.png",
         "prompt": "What animal is in this image and where is it sitting?"}' \
    http://localhost:8081/

curl -X POST -H "Content-Type: application/json" \
    -d '{"image_url": "file://sample_data/dogs_running/dog_running_01.png",
         "prompt": "Describe what the animal is doing in this image."}' \
    http://localhost:8081/
```

```bash
curl -X POST "http://localhost:8082/text_detect" \
  -F "image=@sample_data/cats_on_desk/cat_on_desk_01.png" \
  -F "text=cat desk computer"

curl -X POST "http://localhost:8082/text_detect" \
  -F "image=@sample_data/dogs_running/dog_running_01.png" \
  -F "text=dog field grass"
```


Images generated using:
- **Model**: IDKiro/sdxs-512-dreamshaper
- **Resolution**: 512x512 pixels
- **Inference Steps**: 1
- **Guidance Scale**: 7.5

Total images generated: 1
