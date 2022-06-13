import json
from decomposition import decompose


input_path = 'English_12.json'


output_path = "English_12.json"

if __name__ == '__main__':
    
    with open(input_path) as json_file:
        transcript_json = json.load(json_file)
        
    decomposed_json = decompose(transcript_json)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(decomposed_json, f, ensure_ascii=False)