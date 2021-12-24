import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label # Быстрая маркировка

inner_mask = np.array([[0, 0], [0, 1]])
external_masks = np.array([[0, 1], [1, 1]])


def match(a: np.array, mask: np.array) -> bool:
    if np.all(a == mask):
        return True
    return False


def count_objects(image: np.array) -> tuple:
    e = 0
    i = 0
    for y in range(0, image.shape[0] - 1):
        for x in range(0, image.shape[1] - 1):
            sub = image[y:y+2, x:x+2]
            if match(sub, inner_mask):
                e += 1
                continue
            if match(sub, external_masks):
                i += 1
                continue

    return (e, i)


def euler(labeled, label):
    pos = np.where(labeled == label)

    y_min, y_max = pos[0].min(), pos[0].max()
    x_min, x_max = pos[1].min(), pos[1].max()

    image = np.zeros((y_max-y_min+2) * (x_max-x_min+2))
    image = image.reshape((y_max-y_min+2, x_max-x_min+2))

    image[1:y_max-y_min+1, 1:x_max-x_min +
          1] = labeled[y_min:y_max, x_min:x_max]
    pos = np.where(image == label)
    
    image[pos] = 1

    x, v = count_objects(image)
    return (x, v)


def count_object(labeled):
    objects = {}

    for i in range(1, labeled.max() + 1):
        figures = euler(labeled, i)

        if figures in objects:
            objects[figures] += 1
        else:
            objects[figures] = 1

    return objects


if __name__ == "__main__":
    image = np.load("./data/ps.npy")

    # plt.imshow(image)
    # plt.show()

    labeled = label(image)
    objects = count_object(labeled)

    print(f"Всего найдено {labeled.max()} объектов")

    for i, v in enumerate(objects.values()):
        print(f"[Тип #{i}] Количество объектов: {v}")
