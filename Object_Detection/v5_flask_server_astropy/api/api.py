import json
import requests
from tqdm import tqdm
from .secret import API, get_response

DATABASE = "./api/db.json"

def make_request(api_name, params, headers):
    api_dict = API[api_name]
    api_url = api_dict['url']
    api_auth = api_dict['auth']

    input_params = {}
    input_params.update(api_auth)
    input_params.update(params)
    
    response = requests.get(
        api_url, 
        params=input_params, 
        headers=headers)

    result_dict = get_response(api_name,response)
    return result_dict

def update_db(bird_list, api_name="edamam"):
    query_str = API[api_name]['query_str']
    headers = {"Accept": "application/json",}
    db = []
    for bird_name in tqdm(bird_list):
        query_str['ingr'] = bird_name
        bird_dict = make_request(
            api_name, 
            params=query_str,
            headers=headers)

        if bird_dict is not None:
            db.append(bird_dict)
        else:
            print(f"Failed to get {bird_name}")
    save_db(db)

def save_db(db, out_name=DATABASE):
    # db: list[dict]
    # data: list[dict]

    with open(out_name, 'r') as f:
        data = json.load(f)
    
    data['bird'] += db
    with open(out_name, 'w') as f:
        json.dump(data,f)

def get_info_from_db(bird_list):
    if not isinstance(bird_list, list):
        bird_list = list(bird_list)

    with open(DATABASE, 'r',encoding='utf-8') as f:
        data = json.load(f)
    
    result_list = {
        "中文名": [],
        "别名": [],
        "目": [],
        "科": [],
        "形态特征": [],
        "栖息环境": [],
        "生活习性":[],
        "分布范围":[],
        "现存数量":[],
    }
    for bird in bird_list:
        has_info=False
        for item in data['bird']:
            if bird == item['name']:
                for key in result_list.keys():
                    result_list[key].append(item['introduction'][key])
                has_info = True
                break
        if not has_info:
            for key in result_list.keys():
                result_list[key].append(None)
    print(result_list)
    return result_list


if __name__ == '__main__':
    bird_list = ['fengtouyanou',
                 'huiweiou',
                 'heilianpilu',
                 'huangzuibailu',
                 'douyan',
                 'heizhenyanou',
                 'boat',
                 'tiane',
                 'canglu',
                 'baijiling',
                 'zhongbiaoyu']
    update_db(bird_list)
