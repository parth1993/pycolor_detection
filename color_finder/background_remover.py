import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class BackgroundRemover:
    def __init__(self, image_path, model=None):
        self.image_path = image_path
        self.model = model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Check if U2NET model is provided for advanced removal
        if model is not None:
            self.model.to(self.device)

    def read_image(self):
        # Utility method to read image
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError("Image not found or invalid image path provided.")
        return image

    def simple_removal(self, threshold=240):
        image = self.read_image()
        # Convert to RGB for consistent processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.all(image_rgb > threshold, axis=-1)
        background_removed = np.zeros_like(image_rgb)
        background_removed[~mask] = image_rgb[~mask]
        return cv2.cvtColor(background_removed, cv2.COLOR_RGB2BGR)

    def preprocess_image_for_u2net(self, image):
        preprocess = transforms.Compose(
            [
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return preprocess(image).unsqueeze(0)  # Add batch dimension

    def post_process_u2net(self, mask):
        mask = mask.squeeze().cpu().detach().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        return mask

    def u2net_removal(self):
        if self.model is None:
            raise ValueError("U2NET model not provided.")

        image = Image.open(self.image_path).convert("RGB")
        input_tensor = self.preprocess_image_for_u2net(image).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)[0]
        mask = self.post_process_u2net(output)

        original_image = self.read_image()
        original_image = cv2.resize(
            original_image, (mask.shape[1], mask.shape[0])
        )
        foreground = cv2.bitwise_and(original_image, original_image, mask=mask)
        return foreground

    def remove_background(self, method="simple", threshold=240):
        if method == "simple":
            return self.simple_removal(threshold)
        elif method == "u2net":
            return self.u2net_removal()
        else:
            raise ValueError(
                "Invalid method specified. Use 'simple' or 'u2net'."
            )


# Example usage:
# For U2NET, assume 'u2net_model' is your loaded model.
# image_path = 'path_to_your_image.jpg'
# remover = BackgroundRemover(image_path, model=u2net_model)
# image_without_bg = remover.remove_background(method='u2net') # or 'simple' for simple removal

# To view the result for 'simple', replace 'image_without_bg' with the result of 'simple' removal.
# cv2.imshow('Result', image_without_bg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
