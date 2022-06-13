# Decomposer

after installing the requirements, please run the following lines

```
python3 -m spacy download ru_core_news_sm
python3 -m spacy download en_core_web_sm
```

input json example:
```
{'message_list': [{'end_time': '0:00:01.299000',
   'id': 0,
   'start_time': '0:00:00.079000',
   'text': 'Сразу запись начну кстати'},
  {'end_time': '0:00:03.579000',
   'id': 1,
   'start_time': '0:00:02.540000',
   'text': 'Да сразу'}],
 'recording_id': 'e032073mnog3san85eth'}
 ```
 
input json example:

```
{'message_list': [{'end_time': '0:00:01.299000',
   'id': 0,
   'start_time': '0:00:00.079000',
   'text': 'Сразу запись начну кстати'},
  {'end_time': '0:00:03.579000',
   'id': 1,
   'start_time': '0:00:02.540000',
   'text': 'Да сразу'}],
 'recording_id': 'e032073mnog3san85eth',
 'summary' : 'text',
 'topic' : 'text',
 'task' : 'text',
 'reminder' : 'text',
 'colored' : {"BEEN DONE' : [['black text', 'colored text', 'black text', 'colored text', 'black text'], ['black text', 'colored text', 'black text', 'colored text', 'black text']}, "TODO' : [['black text', 'colored text', 'black text', 'colored text', 'black text'], ['black text', 'colored text', 'black text', 'colored text', 'black text']}}
```

##### *NOTE that every ODD id of the colored text junks should be colored 
