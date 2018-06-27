import json


if __name__ == '__main__':
    # with open('org_captions.json') as f:
    #     data = json.load(f)
    #
    # clean_data = {}
    # for each in data:
    #     clean_data[each] = data[each].replace('.', '').replace('(', '').replace(')', '').split()
    #
    # with open('captions.json', 'w') as f:
    #     json.dump(clean_data, f)
    data = []
    with open("new_test_lst.txt") as f:
        for each in f.readlines():
            data.append("{}.png".format(each.strip()))

    with open("test_files.json", 'w')  as f:
        json.dump(data, f)
