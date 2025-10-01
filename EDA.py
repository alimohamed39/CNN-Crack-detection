import os
import random
import shutil
from PIL import Image
import matplotlib.pyplot as plt



data_set_dir = r"C:\Users\alime\OneDrive - Johannes Kepler Universität Linz\Surface Crack detection proj"

data_split = False #Only to ensure the data split function gets called once
data_count = False
data_plot = False
resize_images = False
get_dim = True


#############################################################
# Check image size
def getsize(filename):
    """Check if we have any images with a size different than 227x227"""
    im = Image.open(filename)
    if im.size != (227, 227):
        print(f"Warning: {filename} has size {im.size}")
    else:
        print("The image has a size of ",Image.open(filename).size)



#############################################################
# Count images
def image_count(n_images, p_images):
    """Gets the number of images we have in the dataset"""

    image_count_N = 0
    image_count_P = 0

    for file in os.listdir(n_images):
        image_count_N += 1
        getsize(os.path.join(n_images, file))

    for file in os.listdir(p_images):
        image_count_P += 1
        getsize(os.path.join(p_images, file))


    return image_count_N, image_count_P


#############################################################
# Split data into train/val/test
def data_split(train_split=0.7, val_split=0.2):
    cls = ["Negative", "Positive"]
    test_split = 1 - (train_split + val_split)

    for cl in cls:
        files = os.listdir(os.path.join(data_set_dir, cl))
        random.shuffle(files)

        n_total = len(files)
        n_train = int(train_split * n_total)
        n_val = int(val_split * n_total)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for f, subset in [(train_files, "train"), (val_files, "val"), (test_files, "test")]:
            folder = os.path.join(data_set_dir, subset, cl)
            os.makedirs(folder, exist_ok=True)
            for file in f:
                shutil.copy(os.path.join(data_set_dir, cl, file), os.path.join(folder, file))


#############################################################
# Plot an image

def plot_random_images(subset="train", class_name="Negative", n=5):
    """
    Show n random images from a given subset (train/val/test) and class (Negative/Positive).
    """
    folder = os.path.join(data_set_dir, subset, class_name)
    files = random.sample(os.listdir(folder), n)

    plt.figure(figsize=(18, 7))
    for i, file in enumerate(files, 1):
        img = Image.open(os.path.join(folder, file))
        plt.subplot(1, n, i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{class_name} ({subset})")
    plt.show()

######################################################

def resize_images(path):
    """
    Resize images in the given folder to 224x224 for ResNet.
    """
    for file_name in os.listdir(path):
        file_image = os.path.join(path, file_name)

        if not file_image.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            image = Image.open(file_image).convert("RGB")  # ensure 3 channels
            image = image.resize((224, 224))

            # Save back with same filename (overwrites original)
            image.save(file_image)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")


#############################################################
# Calling the functions



if data_split:
    data_split(train_split=0.7, val_split=0.2)
if data_count :
    train_n, train_p = image_count(
        os.path.join(data_set_dir, "train", "Negative"),
        os.path.join(data_set_dir, "train", "Positive")
    )
    val_n, val_p = image_count(
        os.path.join(data_set_dir, "val", "Negative"),
        os.path.join(data_set_dir, "val", "Positive")
    )
    test_n, test_p = image_count(
        os.path.join(data_set_dir, "test", "Negative"),
        os.path.join(data_set_dir, "test", "Positive")
    )
    orig_n = train_n + val_n + test_n
    orig_p = train_p + val_p + test_p


    print("Training count is", train_n, "for negative and", train_p, "for positive images")
    print("Validation count is", val_n, "for negative and", val_p, "for positive images")
    print("Test count is", test_n, "for negative and", test_p, "for positive images")
    print("The whole dataset count  is", image_count(os.path.join(data_set_dir, "Negative"), os.path.join(data_set_dir,  "Positive")))

    assert train_n + val_n + test_n == orig_n, "Mismatch in Negative split counts!"
    assert train_p + val_p + test_p == orig_p, "Mismatch in Positive split counts!"


if data_plot:
    plot_random_images(subset="train", class_name="Negative",n=10)
    plot_random_images(subset="train", class_name="Positive",n=10)



if resize_images:
    resize_images(os.path.join(data_set_dir,"train", "Negative"))
    resize_images(os.path.join(data_set_dir, "train", "Positive"))
    resize_images(os.path.join(data_set_dir, "val", "Negative"))
    resize_images(os.path.join(data_set_dir, "val", "Positive"))
    resize_images(os.path.join(data_set_dir, "test", "Negative"))
    resize_images(os.path.join(data_set_dir, "test", "Positive"))

if get_dim:
    def get_size(path):
        for file in os.listdir(path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                file_path = os.path.join(path, file)
                with Image.open(file_path) as img:
                    print(f"Image{file} size: {img.size}")


    get_size(os.path.join(data_set_dir, "train", "Negative"))

