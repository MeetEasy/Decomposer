import json
from decomposition import decompose

import os
dir = os.path.dirname(__file__)


input_path = os.path.join(dir, '12.json')


output_path = os.path.join(dir, f'{os.path.basename(input_path)}.json')

if __name__ == '__main__':
    
    with open(file_path) as json_file:
        transcript_json = json.load(input_path)
        
    decomposed_json = decompose(transcript_json)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summarized_json, f, ensure_ascii=False)