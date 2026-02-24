import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
model.eval()

# Body part prompts
labels = [
    "a chest x-ray radiograph showing lungs and ribs",
    "a skull x-ray radiograph showing head bones",
    "a dental x-ray radiograph showing teeth",
    "a limb x-ray radiograph showing arm or leg bones"
]

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)

    probs = probs[0].cpu().numpy()

    print(f"\nResults for {image_path}:")
    for label, prob in zip(labels, probs):
        print(f"{label}: {prob:.4f}")

    predicted_index = probs.argmax()
    predicted_label = labels[predicted_index]

    print(f"→ Predicted class: {predicted_label}")
    print("-" * 50)


# Add your images here
images = [
    "pic1.jpg",
    "pic2.jpg",
    "pic3.jpg",
    "pic4.jpg"
]

for img in images:
    classify_image(img)
