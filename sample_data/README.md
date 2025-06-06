# Sample Data

This directory contains sample images generated using the SDXS-512-Dreamshaper model for testing the Ray Serve ML inference services.


- `cats_on_desk/cat_on_desk_01.png` - Cat sitting on office desk variation 1
- `cats_on_desk/cat_on_desk_02.png` - Cat sitting on office desk variation 2
- `cats_on_desk/cat_on_desk_03.png` - Cat sitting on office desk variation 3
- `cats_on_desk/cat_on_desk_04.png` - Cat sitting on office desk variation 4
- `cats_on_desk/cat_on_desk_05.png` - Cat sitting on office desk variation 5

### Dogs Running in Fields
- `dogs_running/dog_running_01.png` - Dog running in field variation 1
- `dogs_running/dog_running_02.png` - Dog running in field variation 2
- `dogs_running/dog_running_03.png` - Dog running in field variation 3
- `dogs_running/dog_running_04.png` - Dog running in field variation 4
- `dogs_running/dog_running_05.png` - Dog running in field variation 5


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
- **Guidance Scale**: 0

Total images that will be generated: 10
