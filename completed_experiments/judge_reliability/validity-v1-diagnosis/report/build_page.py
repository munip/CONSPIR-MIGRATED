import json
T = str(__import__('pathlib').Path(__file__).resolve().parent) + '/'
data = json.load(open(T + 'data.json'))
page = open(T + 'template.html').read().replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
open(T + 'index.html', 'w').write(page)
print(len(page), 'bytes')
