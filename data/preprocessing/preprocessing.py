import torch
import torchvision.transforms as T

input_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class Augmentor:
    def __init__(self, chances: dict):
        self.chances = chances

        self.rotate = T.RandomRotation(2)
        self.blur = T.GaussianBlur(3, (0.1, 1.0))
        self.colour_jitter = T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.1,
            hue=0.02,
        )
        self.crop = T.RandomResizedCrop(
            224,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1)
        )

    def __call__(self, image):
        if torch.rand(1) < self.chances.get("crop", 0.0):
            image = self.crop(image)

        if torch.rand(1) < self.chances.get("rotation", 0.0):
            image = self.rotate(image)

        if torch.rand(1) < self.chances.get("blur", 0.0):
            image = self.blur(image)

        if torch.rand(1) < self.chances.get("colour_jitter", 0.0):
            image = self.colour_jitter(image)

        return image

chances = { 
    "crop": 0.15, 
    "rotation": 0.25, 
    "blur": 0.3, 
    "colour_jitter": 0.35, 
}

augment = Augmentor(chances)

def preprocess(image):
    image = augment(image)
    image = input_transform(image)
    return image
