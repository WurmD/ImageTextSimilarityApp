import falcon

from detect_objects_in_image_compute_similarity_with_text import ImageTextSimilarity

api = application = falcon.API()
image_text_similarity_object = ImageTextSimilarity()
api.add_route('/image_text_similarity', image_text_similarity_object)
