import json

info = {
        "description": "COCO 2017 Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2017,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    }

data = {'info':info, 'licenses':[], 'images':[], 'annotations':[], 'categories':[]}

data["licenses"].append(
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    }
)

data['categories'].append({
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [
            "nose"
        ],
        "skeleton": []
    })

with open('test.json', 'w') as file:
    json.dump(data, file)