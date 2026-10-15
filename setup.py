import os
import sys
from PIL import Image

def main():
    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Directory '{images_dir}' is ready.")
    print("To take a screenshot, you can use tools like `scrot`, `gnome-screenshot`, or place an image into the `images/` folder.")
    print("Example: save your screenshot as `images/test.jpg`")

if __name__ == "__main__":
    main()
