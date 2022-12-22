import numpy as np
import torch

class ToMask(object):
    """
        Convert a 3 channel RGB image into a 1 channel segmentation mask
    """
    def __init__(self, palette_dictionnary):
        self.nb_classes = len(palette_dictionnary)
        # sort the dictionary of the classes by the sum of rgb value -> to have always background = 0
        # self.converted_dictionnary = {i: v for i, (k, v) in enumerate(sorted(palette_dictionnary.items(), key=lambda item: sum(item[1])))}
        # import ipdb; ipdb.set_trace()
        self.palette_dictionnary = palette_dictionnary

    def __call__(self, pil_image):
        # avoid taking the alpha channel
        image_array = np.array(pil_image)
        if len(image_array.shape) == 3:
            image_array = image_array[:, :, :3]
            output_array = np.zeros(image_array.shape, dtype="int")[:, :, 0]
            for label in self.palette_dictionnary.keys():
                rgb_color = self.palette_dictionnary[label]['color']
                mask = (image_array == rgb_color)
                output_array[mask[:, :, 0]] = int(label)
        else:
            output_array = image_array.astype("int")
            # print(np.unique(output_array))



        output_array = torch.from_numpy(output_array).unsqueeze(0).long()
        return output_array
