import xml.etree.ElementTree as ET
tree = ET.parse('beethoven_extracted.xml')
root = tree.getroot()

data_list = []
for data_set in root.findall('.//data_set'):
    data_point = {
        'data_set_id': data_set.find('data_set_id').text,
    }
    for feature in data_set.findall('feature'):
        name = feature.find('name').text
        value = feature.find('v').text
        data_point[name] = value
    data_list.append(data_point)
import pandas as pd
df = pd.DataFrame(data_list)
df = df.apply(lambda x: x.str.replace(',', '.').str.replace('E', 'e', case=False))

df.to_csv('cleaned_data.csv', index=False,sep = ";")

