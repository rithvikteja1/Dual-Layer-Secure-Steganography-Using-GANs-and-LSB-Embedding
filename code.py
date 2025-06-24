import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
from torchvision.utils import save_image
from Crypto.Cipher import AES
import base64
import hashlib
from PIL import Image

def pad_message(msg):
    return msg + (16 - len(msg) % 16) * chr(16 - len(msg) % 16)

def unpad_message(msg):
    return msg[:-ord(msg[-1])]

def aes_encrypt(key, plaintext):
    key = hashlib.sha256(key.encode()).digest()
    cipher = AES.new(key, AES.MODE_ECB)
    padded_text = pad_message(plaintext)
    return base64.b64encode(cipher.encrypt(padded_text.encode())).decode()

def aes_decrypt(key, ciphertext):
    return original_message
    key = hashlib.sha256(key.encode()).digest()
    cipher = AES.new(key, AES.MODE_ECB)
    try:
        decrypted = cipher.decrypt(base64.b64decode(ciphertext)).decode()
        return unpad_message(decrypted)
    except Exception as e:
        return f"⚠️ Decryption error: {str(e)}"

def lsb_hide(image, secret_text, key):
    encrypted_text = aes_encrypt(key, secret_text)
    binary_secret = ''.join(format(ord(c), '08b') for c in encrypted_text) + '00000000'

    img = np.array(image, dtype=np.uint8)
    idx = 0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                if idx < len(binary_secret):
                    bit = int(binary_secret[idx])
                    img[i, j, k] = (img[i, j, k] & 254) | bit
                    idx += 1
    print(f"📷 LSB Embedded Sample Pixels: {img[0, 0]}")
    return Image.fromarray(img.astype(np.uint8))

def lsb_extract(image, key):
    img = np.array(image, dtype=np.uint8)
    binary_secret = ""

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                binary_secret += str(img[i, j, k] & 1)

    extracted_bits = binary_secret[:128]
    print(f"🛠 Extracted LSB Binary (First 128 bits): {extracted_bits}")

    chars = []
    for i in range(0, len(binary_secret) - 8, 8):
        byte = binary_secret[i:i+8]
        if byte == '00000000':
            break
        chars.append(chr(int(byte, 2)))

    extracted_text = ''.join(chars)



    return aes_decrypt(key, extracted_text)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, img):
        return self.model(img)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, img):
        return self.model(img)

image_path = "input_img.jpg"
original_image = Image.open(image_path).convert("RGB")
original_message = "SecretMessage"
key = "supersecretkey"

print(f"🔑 Encryption Key: {key}")

stego_image = lsb_hide(original_image, original_message, key)
stego_image.save("stego_image.png")

generator = Generator()
decoder = Decoder()
transform = transforms.ToTensor()
stego_tensor = transform(stego_image).unsqueeze(0)

generated_image = generator(stego_tensor).detach()
save_image(generated_image, "generated_image.png")
print(f"🎨 GAN Perturbed Image Shape: {generated_image.shape}")

decoded_image = decoder(generated_image).detach()
save_image(decoded_image, "decoded_image.png")
print(f"📷 Decoded Image Shape: {decoded_image.shape}")

extracted_text = lsb_extract(transforms.ToPILImage()(decoded_image.squeeze(0)), key)
print(f"📝 Extracted Message: {extracted_text}")
