import pandas as pd
import requests
from io import StringIO

orig_url='https://drive.google.com/file/d/1-tEsH5W-WBO3mXydvrxhO9TSAiQa2ZYM/view?usp=sharing'

file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/u/0/uc?export=download&confirm=_KGE&id=1-tEsH5W-WBO3mXydvrxhO9TSAiQa2ZYM'
url = requests.get(dwn_url).text
csv_raw = StringIO(url)
dfs = pd.read_csv(csv_raw)




import pandas as pd

url = 'https://drive.google.com/file/d/1-tEsH5W-WBO3mXydvrxhO9TSAiQa2ZYM/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)
print(df.columns)


