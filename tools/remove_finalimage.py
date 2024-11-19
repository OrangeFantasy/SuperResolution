import os

root = "E:/SuperSampling/Bunker/NoAA_Temp"

for file in os.listdir(root):
    if file.endswith(".FinalImage.exr"):
        print(file)
        os.remove(os.path.join(root, file))
