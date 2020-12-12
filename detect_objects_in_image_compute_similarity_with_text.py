import argparse
import copy
import cv2
import falcon
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
    image_text_similarity_object = ImageTextSimilarity(args)

    if args.image_path:
        image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
        similarity_value, output_labels = image_text_similarity_object.detect_objects_in_image_compute_textual_similarity(image, args.textual_description)
        print(args.image_path + '[' + output_labels[:-1] + '] x', args.textual_description, '=', similarity_value)
    else:
        api = application = falcon.API()
        api.req_options.auto_parse_form_urlencoded = True
        api.add_route('/image_text_similarity', image_text_similarity_object)



def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute similarity between image content and text description")
    parser.add_argument("--config_path", default='deeplabpytorch/configs/cocostuff164k.yaml', type=str)
    parser.add_argument("--model_path", default='deeplabpytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth', type=str)
    parser.add_argument("--image_path", default='', type=str)
    parser.add_argument("--textual_description", default='', type=str)
    args = parser.parse_args()

    if (args.model_path == 'deeplabpytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth'
        and not os.path.exists(args.model_path)):
        wget.download('https://www.dropbox.com/s/icpi6hqkendxk0m/deeplabv2_resnet101_msc-cocostuff164k-100000.pth?raw=1',
            out='deeplabpytorch/')

    return args

class ImageTextSimilarity():
    def __init__(self, args, print_object_masked_images=False):
        self.print_object_masked_images = print_object_masked_images
        self.setup_model_device_config_and_labels(args)

    def setup_model_device_config_and_labels(self, args):
        self.CONFIG = OmegaConf.load(args.config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)

        self.model = eval(self.CONFIG.MODEL.NAME)(n_classes=self.CONFIG.DATASET.N_CLASSES)
        state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        label_file = self.CONFIG.DATASET.LABELS
        self.labels_no_spaces_dictionary = {}
        with open(label_file) as f:
            for line in f:
                key = line.split()[0]
                val = ''.join(line.split()[1:])
                self.labels_no_spaces_dictionary[int(key)] = val
        # print(labels_no_spaces_dictionary)

        # return model, device, CONFIG, labels_no_spaces_dictionary

    def on_post(self, req, resp):
        image = req.get_param("image", required=True)
        print(image) # debug
        text = req.get_param("text", required=True)
        similarity, output_labels = self.detect_objects_in_image_compute_textual_similarity(image, text)

        payload = {}
        payload['similarity'] = similarity
        payload['debug_similarity'] = 'image[' + output_labels[:-1] + '] x ' +  text +  ' = ' + str(similarity)

        resp.status = falcon.HTTP_200


    def detect_objects_in_image_compute_textual_similarity(self, image, text):
        image, raw_image = preprocessing(image, self.device, self.CONFIG)
        labelmap = inference(self.model, image, raw_image)
        labels = np.unique(labelmap)
        
        original_image = raw_image[:,:,::-1]
        # imsave('original.png', original_image)
        output_labels = ''
        for i, label in enumerate(labels):
            mask = labelmap == label
            area = sum(sum(mask))
            if area < mask.shape[0] * mask.shape[1] / 100:
                continue
            output_labels += self.labels_no_spaces_dictionary[label] + ' '

            if self.print_object_masked_images:
                notmask = labelmap != label
                this_label_image = copy.deepcopy(original_image)
                this_label_image[notmask] = 0
                imsave('area' + str(area) + '_' + self.labels_no_spaces_dictionary[label] + '[' + str(label) +'].png', this_label_image, check_contrast=False)

        similarity_value = cosine_sim(output_labels, text)

        return similarity_value, output_labels

if __name__ == "__main__":
    main()