import torchvision.transforms as T

input_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

class Augmentor:
    def __init__(self, chances: dict):
        self.chances = chances
        self.rotate = T.RandomApply(
            [T.RandomRotation(10)],
            p=self.chances.get("rotation", 0.0)
        )
        self.crop = T.RandomApply(
            [T.RandomResizedCrop(
            size=500,
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05),
            interpolation=T.InterpolationMode.BILINEAR,
            )],
            p=self.chances.get("crop", 0.0)
        )
        self.blur = T.RandomApply(
            [T.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0)
            )],
            p=self.chances.get("blur", 0.0)
        )
        
    def __call__(self, image):
        # cropping will only change how the final resize looks
        # we have to think about whether we want this
        # image = self.crop(image)
        image = self.rotate(image)
        image = self.blur(image)
        
        return image
    
chances = {
    "rotation": 0.3,
    "crop": 0.3,
    "blur": 0.3
}
    
augment = Augmentor(chances)
        

def input_processing(image):
    # getting the image ready to be input for a ResNet
    return input_transform(image)

def augmentation(image):
    return augment(image)

def preprocess(image):    
    image = augmentation(image)
    
    image = input_processing(image)
    
    return image
    