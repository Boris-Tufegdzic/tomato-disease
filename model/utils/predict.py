import torchvision
import torch
from typing import List, Tuple

from PIL import Image
import torchvision.transforms as transforms


#device definition
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred(model: torch.nn.Module,
         image_path: str,
         image_size: Tuple[int, int] = (224, 224),
         transform: torchvision.transforms = None,
         device: torch.device = device):
    
    # 2. Open image
    try:
        img = Image.open(image_path)
    except Exception as e:
        return jsonify({'error': f'Failed to load image: {str(e)}'}), 500


    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 
    
    # 4. Verification the model is on the target device
    model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    
    with torch.inference_mode():
        # 6. Transform and add an extra dimension to image, to match model dimensions requirement : [batch_size, color_channels, height, width]
        transformed_image = image_transform(img).unsqueeze(dim=0)
        

        # 7. prediction on image with an extra dimension (sent to the target device)
        target_image_pred = model(transformed_image.to(device))
        

    # 8. Conversion logits -> prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Conversion prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    return target_image_pred_label
