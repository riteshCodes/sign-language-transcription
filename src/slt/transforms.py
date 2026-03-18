from torchvision import transforms

# Parameters for transformation
CHANNEL = 3
IMAGE_SIZE = 256
CENTER_CROP_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RANDOM_ROTATION = 10  # Degree
RANDOM_HORIZONTAL_FLIP = 0.5  # Probability

# ToTensor() : Transforms PIL Image/ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]

train_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                      transforms.CenterCrop(CENTER_CROP_SIZE),
                                      transforms.RandomHorizontalFlip(
                                          RANDOM_HORIZONTAL_FLIP),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=MEAN, std=STD)])


test_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                     transforms.CenterCrop(CENTER_CROP_SIZE),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=MEAN, std=STD)])
