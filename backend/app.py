import numpy as np
import os

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)
#MNIST files are binarty files, so we need to read them in binary mode
#magic = file type, num = number of images, rows = number of rows per image 28, cols = number of columns per image 28
#at first we read the first 16 bytes to get the header information, then we read the rest of the file to get the image data
#we return finally the data reshaped to (num, rows, cols)

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data
#reads the first 8 bytes to get the header of the label file
#magic = file type, num = number of labels
#then we read the rest of the file as uint8
#we return the data as a 1D array of labels

if __name__ == "__main__":
    base = r"C:/Users/Nina/Desktop/Detection/backend/dataset"
    print("Files:", os.listdir(base))

    train_images = load_images(os.path.join(base, "train-images.idx3-ubyte"))
    train_labels = load_labels(os.path.join(base, "train-labels.idx1-ubyte"))
    test_images  = load_images(os.path.join(base, "t10k-images.idx3-ubyte"))
    test_labels  = load_labels(os.path.join(base, "t10k-labels.idx1-ubyte"))
    #load all MNIST files to numpy arrays

    print("Train images:", train_images.shape) #60k images 28x28
    print("Train labels:", train_labels.shape) #60k lables, one per image
    print("Test images:", test_images.shape) #10k test images 28x28
    print("Test labels:", test_labels.shape) #10k text lables, one per image
    
