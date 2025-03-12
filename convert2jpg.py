import os

import cv2


if __name__ == "__main__":
    path2images = "data/single_person"
    new_path = "data/single_person/jpg"
    os.makedirs(new_path, exist_ok=True)
    image_names = [x for x in os.listdir(path2images) if x.endswith(".png")]
    for file in image_names:
        print(file)
        img = cv2.imread(os.path.join(path2images, file))
        cv2.imwrite(os.path.join(new_path, file.replace(".png", ".jpg")), img) 

