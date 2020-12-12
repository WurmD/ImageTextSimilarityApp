import argparse
import copy
import cv2
import numpy as np
from omegaconf import OmegaConf
import os
from skimage.io import imsave
import torch
import wget

from cosine_similarity_between_2_texts import cosine_sim
from deeplabpytorch.demo import inference, preprocessing
from deeplabpytorch.libs.models import DeepLabV2_ResNet101_MSC


def main():
    args = parse_arguments()
    model, device, CONFIG, labels_no_spaces_dictionary = setup_model_device_config_and_labels(args)

    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    image, raw_image = preprocessing(image, device, CONFIG)
    labelmap = inference(model, image, raw_image)
    labels = np.unique(labelmap)
    
    original_image = raw_image[:,:,::-1]
    imsave('original.png', original_image)
    output_labels = ''
    for i, label in enumerate(labels):
        mask = labelmap == label
        area = sum(sum(mask))
        if area < mask.shape[0] * mask.shape[1] / 100:
            continue
        output_labels += labels_no_spaces_dictionary[label] + ' '
        notmask = labelmap != label
        this_label_image = copy.deepcopy(original_image)
        this_label_image[notmask] = 0
        imsave(args.image_path + '_area' + str(area) + '_' + labels_no_spaces_dictionary[label] + '[' + str(label) +'].png', this_label_image, check_contrast=False)

    similarity_value = cosine_sim(output_labels, args.textual_description)
    print(args.image_path + '[' + output_labels[:-1] + '] x', args.textual_description, '=', similarity_value)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute similarity between image content and text description")
    parser.add_argument("--config_path", default='configs/cocostuff164k.yaml', type=str)
    parser.add_argument("--model_path", default='deeplab_pytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth', type=str)
    parser.add_argument("--image_path", default='Djur_034.jpg', type=str)
    parser.add_argument("--textual_description", default='Blue dragonfly on tree branch', type=str)
    args = parser.parse_args()

    if (args.model_path == 'deeplab_pytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth'
        and not os.path.exists(args.model_path)):
        wget.download('https://www.dropbox.com/s/icpi6hqkendxk0m/deeplabv2_resnet101_msc-cocostuff164k-100000.pth?raw=1',
            out='deeplabpytorch/')

    return args

def setup_model_device_config_and_labels(args):
    CONFIG = OmegaConf.load(args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    label_file = CONFIG.DATASET.LABELS
    labels_no_spaces_dictionary = {}
    with open(label_file) as f:
        for line in f:
            key = line.split()[0]
            val = ''.join(line.split()[1:])
            labels_no_spaces_dictionary[int(key)] = val
    # print(labels_no_spaces_dictionary)

    return model, device, CONFIG, labels_no_spaces_dictionary

if __name__ == "__main__":
    main()