import os

pure        = len(os.listdir('dataset/pure'))
adulterated = len(os.listdir('dataset/adulterated'))

print(f"Pure:        {pure} images")
print(f"Adulterated: {adulterated} images")