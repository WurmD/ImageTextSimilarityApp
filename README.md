# ImageTextSimilarityApp

## Requirements

- docker (tested on docker 19.03 in Ubuntu 18.04)
- gpu

## To build

`docker build --tag image_text_similarity_image .`

## To run

- In one terminal:
`docker run --net="host" --gpus=all  --name image_text_similarity_container -ti image_text_similarity_image:latest gunicorn image_similarity_app` which will start up the service, exposing http://127.0.0.1:8000/image_text_similarity as the API endpoing
- In another terminal:
`(echo -n '{"image": "'; base64 Djur_034.jpg; echo '"}') | curl -i -H "Content-Type: application/json" -d @- "http://127.0.0.1:8000/image_text_similarity?text=Blue%20insect%20on%20tree%20branch"`
to send image Djur_034.jpg and text 'Blue insect on tree branch' to the created endpoint http://127.0.0.1:8000/image_text_similarity