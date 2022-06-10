import re

def process_json(transcript_json):
    text= ''

    for message in transcript_json['message_list']:
        text+=message['text']+'. '
    
    return text