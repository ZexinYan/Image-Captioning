import json


if __name__ == '__main__':
    with open('debugging_captions.json', 'r') as f:
        data = json.load(f)
    for each in data:
        data[each] = data[each].replace('.', '').split()
    with open('debugging_captions.json', 'w') as f:
        json.dump(data, f)
