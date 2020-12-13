import argparse
import base64
import copy
import cv2
import datetime
import falcon
from io import BytesIO
import json
import numpy as np
from omegaconf import OmegaConf
import os
from PIL import Image, ImageOps
from skimage.io import imsave
import torch
import wget

from cosine_similarity_between_2_texts import cosine_sim
from deeplabpytorch.demo import inference, preprocessing
from deeplabpytorch.libs.models import DeepLabV2_ResNet101_MSC

def main():
    args = parse_arguments()
    image_text_similarity_object = ImageTextSimilarity(config_path=args.config_path, model_path=args.model_path)

    if args.image_path:
        image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
        similarity_value, output_labels = image_text_similarity_object.detect_objects_in_image_compute_textual_similarity(image, args.textual_description)

    if args.imagedict_base64encoded:
        imagedict_base64encoded = json.loads(args.imagedict_base64encoded)
        nparrayimage = decode_image(imagedict_base64encoded)
        similarity_value, output_labels = image_text_similarity_object.detect_objects_in_image_compute_textual_similarity(nparrayimage, args.textual_description)

    print(args.image_path + '[' + output_labels[:-1] + '] x', args.textual_description, '=', similarity_value)

    # Apparently gunicorn can't read the application in here, inside a function
    # else:
    #     api = application = falcon.API()
    #     api.req_options.auto_parse_form_urlencoded = True
    #     api.add_route('/image_text_similarity', image_text_similarity_object)



def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute similarity between image content and text description")
    parser.add_argument("--config_path", default='deeplabpytorch/configs/cocostuff164k.yaml', type=str)
    parser.add_argument("--model_path", default='deeplabpytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth', type=str)
    parser.add_argument("--image_path", default='', type=str)
    parser.add_argument("--textual_description", default='', type=str)
    parser.add_argument("--imagedict_base64encoded", type=str)
    args = parser.parse_args()

    if (args.model_path == 'deeplabpytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth'
        and not os.path.exists(args.model_path)):
        wget.download('https://www.dropbox.com/s/icpi6hqkendxk0m/deeplabv2_resnet101_msc-cocostuff164k-100000.pth?raw=1',
            out='deeplabpytorch/')

    return args

class ImageTextSimilarity():
    def __init__(self, config_path='deeplabpytorch/configs/cocostuff164k.yaml',
        model_path='deeplabpytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth', print_object_masked_images=False
    ):
        self.print_object_masked_images = print_object_masked_images
        self.setup_model_device_config_and_labels(config_path, model_path)

    def setup_model_device_config_and_labels(self, config_path, model_path):
        self.CONFIG = OmegaConf.load(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)

        self.model = eval(self.CONFIG.MODEL.NAME)(n_classes=self.CONFIG.DATASET.N_CLASSES)
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
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

    def on_put(self, req, resp):
        print('req', req)

        text = req.get_param("text", required=True)
        print('text:', text)

        # Encode image with base64 to send with 
        # (echo -n '{"image": "'; base64 four_test.png; echo '"}') | curl -i -H "Content-Type: application/json" -d @- "http://127.0.0.1:8000/image_text_similarity?text=rabo"
        # add   -X PUT   to make it PUT instead of POST?
        # -i, --include: Include the HTTP response headers in the output
        # -H, --header <header/@file>: Extra header to include in the request when sending HTTP to a server
        # -d, --data <data>: if @- reads from stdin   -- MIGHT WORK ONLY WITH A POST REQUEST
        image_base64encoded = json.loads(req.stream.read())
        print('image_base64encoded:',image_base64encoded)
        decoded_image = base64.b64decode(image_base64encoded.get('image'))
        print('decoded_image:',decoded_image)

        resp = self.compute_similarity_and_repond(resp, decoded_image, text)


    def on_post(self, req, resp):
        print('req', req)

        # pickle.dump(req, clean_string(str(req))) # req is not serializable

        # image_path = req.get_param("image_path")
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # image_name = req.get_param("image_name")
        # image_name = '' if image_name == None else image_name

        # FAIL
        # image_raw = json.loads(req.stream.read())
        # decoded_image = base64.b64decode(image_raw.get('image'))
        # image = convert_image(BytesIO(decoded_image))

        # FAIL
        # image = json.loads(image_stream.stream.read())
        # decoded_image = base64.b64decode(image.get('image'))
        # converted_image = convert_image(BytesIO(decoded_image))

        # https://stackoverflow.com/questions/59689859/how-to-save-an-image-from-post-request-with-falcon-python
        # FAIL - req.get_param("file") returns None; Postman hangs trying to send with image in body as binary, and
        # headers Content-Type: multipart/form-data, boundary: random
        #
        # refimages_path = "/"
        # incoming_file = req.get_param("file")  # get incoming file
        # print('incoming_file:', incoming_file)
        # print('incoming_file.filename:',incoming_file.filename)
        # imgId = str(int(datetime.datetime.now().timestamp() * 1000))
        # filename = imgId + "." + incoming_file.filename.split(".")[-1]


        # print('image', image) # debug
        text = req.get_param("text", required=True)
        print('req.get_param text:', text)

        imagedict_base64encoded = json.loads(req.stream.read())
        nparrayimage = decode_image(imagedict_base64encoded)
        imsave('received_image.png', nparrayimage, check_contrast=False)
        
        resp = self.compute_similarity_and_repond(resp, nparrayimage, text)


    def compute_similarity_and_repond(self, resp, image, text):
        similarity, output_labels = self.detect_objects_in_image_compute_textual_similarity(image, text)
        return self.construct_response(resp, similarity, text, output_labels)

    def construct_response(self, resp, similarity, text, output_labels=None):

        payload = {}
        payload['similarity'] = similarity
        if output_labels:
            payload['debug_similarity'] = 'image[' + output_labels[:-1] + '] x ' +  text +  ' = ' + str(similarity)

        resp.body = json.dumps(payload)
        resp.status = falcon.HTTP_200

        return resp

    def detect_objects_in_image_compute_textual_similarity(self, nparrayimage, text):
        nparrayimage, raw_image = preprocessing(nparrayimage, self.device, self.CONFIG)
        labelmap = inference(self.model, nparrayimage, raw_image)
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

def clean_string(string):
    string = string.replace(':', '_')
    string = string.replace('/', '_')
    string = string.replace(' ', '_')
    string = string.replace('<', '_')
    string = string.replace('>', '_')
    string = string.replace("'", '_')
    string = string.replace('?', '_')
    string = string.replace('=', '_')
    return string


def decode_image(image_base64encoded):
    print('## image_base64encoded:', image_base64encoded)
    decoded_image = base64.b64decode(image_base64encoded.get('image'))
    print('== decoded_image:', decoded_image)
    PILimg = Image.open(BytesIO(decoded_image)).convert('L')
    print('-- PILimg:', PILimg)
    return np.asarray(PILimg)

def convert_image(image):
    img = Image.open(image).convert('L')
    inverted_img = ImageOps.invert(img)
    data = np.asarray(inverted_img, dtype='int32')
    rescaled_data = (data / 255).reshape(1, 28, 28, 1)
    return rescaled_data


if __name__ == "__main__":
    main()
