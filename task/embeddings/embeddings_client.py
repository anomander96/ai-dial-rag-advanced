import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


class DialEmbeddingsClient:
    
    def __init__(self, deployment_name,  api_key): 
        self.deployment_name = deployment_name
        self.api_key = api_key

    def get_embeddings(self, input_texts: list[str], dimensions: int = 1536) -> dict:
        url = DIAL_EMBEDDINGS.format(model = self.deployment_name)
    
        headers = {
            'Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }
    
        body = {
            'input': input_texts,
            'dimensions': dimensions
        }
    
        response = requests.post(url = url, headers = headers, json = body)
        response.raise_for_status()
        response_json = response.json()
    


        return {
            item['index'] : item['embedding']
            for item in response_json['data']
        }


# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
