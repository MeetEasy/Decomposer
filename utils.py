import re
import pymorphy2
import random
from keybert import KeyBERT
from collections import defaultdict

kw_model = KeyBERT()
morph = pymorphy2.MorphAnalyzer()


def process_json(transcript_json):
    
    if 'text' in transcript_json.keys():
        return transcript_json['text']
        
    else:
        text = ''
        speaker = "SPEAKER_00"

        if "speaker" in transcript_json['message_list'][0].keys():
            for message in transcript_json['message_list']:
                if message['speaker'] == speaker:
                    text = text[:-1]
                    text += ' '+message['text'].lower()+' '
                else:
                    text = text[:-1]
                    text += '. '+message['text']+'.'
                    speaker = message['speaker']

        else:
            for message in transcript_json['message_list']:
                text += message['text'].lower() + ' '
        return text


def split_text_by_speaker(transcript_json):
    
    texts_by_speaker = {}
    speaker = "SPEAKER_00"
    
    if "speaker" in transcript_json['message_list'][0].keys():
    
        for message in transcript_json['message_list']:
            if message['speaker'] in texts_by_speaker.keys():
                if message['speaker']==speaker:

                    texts_by_speaker[message['speaker']] = texts_by_speaker[message['speaker']][:-1]
                    texts_by_speaker[message['speaker']]+=' '+message['text'][0].lower()+message['text'][1:]+' '
                else:
                    texts_by_speaker[message['speaker']] = texts_by_speaker[message['speaker']][:-1]
                    texts_by_speaker[message['speaker']]+='. '+message['text']+'.'
                    speaker = message['speaker']
            else:
                texts_by_speaker[message['speaker']]=message['text']+'.'
                if message['speaker']!=speaker:
                    speaker = message['speaker']
    else:
        texts_by_speaker[speaker] = process_json(transcript_json)
                
    return texts_by_speaker


def get_tasks(transcript_json, doc, nlp, dep_matcher, dep_matches):
    
    text = process_json(transcript_json)

    tasks = []

    for i, match in enumerate(dep_matches):
        pattern_name = match[0]
        matches = match[1]

        if nlp.vocab[pattern_name].text in ['task', 'need', 'want',"could_you"]:
            
            tasks.append(doc[matches[0]].text+' '+join_dependant_tokens(1, doc, matches))
            
        if nlp.vocab[pattern_name].text == 'strong_do':
            
            tasks.append(doc[matches[0]].text+' '+join_dependant_tokens(3, doc, matches))
            
#             output = sorted([matches[0], matches[1]])
#             tasks.append(["..." + doc[output[0] - 6: output[0]].text, doc[output[0]].text,
#                                  doc[output[0]+1:output[1]].text, doc[output[1]].text, doc[output[1]+1: output[1] + 15].text+"..."])

#             tasks.append(join_dependant_tokens(1, doc, matches))

#         elif nlp.vocab[pattern_name].text == 'strong_do':

#             tasks.append(morph.parse(doc[matches[0]].text)[
#                          0].normal_form+' '+join_dependant_tokens(3, doc, matches))

    return list(set(tasks))


def get_personal_tasks(transcript_json, doc, nlp, dep_matcher, dep_matches):
    
    tasks_by_speaker = {}
    
    texts_by_speaker = split_text_by_speaker(transcript_json)
    
    for key in texts_by_speaker.keys():
        
        tasks = []
        doc = nlp(texts_by_speaker[key])
        dep_matches = dep_matcher(doc)
        
        for i, match in enumerate(dep_matches):
            pattern_name = match[0]
            matches = match[1]
            
            if nlp.vocab[pattern_name].text in ['task','need']:
                
                output = sorted([matches[0], matches[1]])
                tasks.append(["..." + doc[output[0] - 6: output[0]].text, doc[output[0]].text,
                                 doc[output[0]+1:output[1]].text, doc[output[1]].text, doc[output[1]+1: output[1] + 15].text+"..."])
                
#                 tasks.append(join_dependant_tokens(1, doc, matches))
    
#             elif nlp.vocab[pattern_name].text == 'strong_do':

#                 tasks.append(morph.parse(doc[matches[0]].text)[0].normal_form+' '+join_dependant_tokens(3, doc, matches))

        tasks_by_speaker[key] = tasks
        
    return tasks_by_speaker


def get_en_tasks(transcript_json, nlp, dep_matcher):
    
    tasks_by_speaker = {}
    
    texts_by_speaker = split_text_by_speaker(transcript_json)
    
    for key in texts_by_speaker.keys():
        
        tasks = []
        doc = nlp(texts_by_speaker[key])
        dep_matches = dep_matcher(doc)
        
        for i, match in enumerate(dep_matches):
            pattern_name = match[0]
            matches = match[1]
            
            if nlp.vocab[pattern_name].text in ['task','need', "could_you"]:
                
                output = sorted([matches[0], matches[1]])
                tasks.append(["..." + doc[output[0] - 6: output[0]].text, doc[output[0]].text,
                                 doc[output[0]+1:output[1]].text, doc[output[1]].text, doc[output[1]+1: output[1] + 15].text+"..."])
                
#                 tasks.append(join_dependant_tokens(1, doc, matches))
    
#             elif nlp.vocab[pattern_name].text == 'strong_do':

#                 tasks.append(doc[matches[0]].text+' ' +
#                          join_dependant_tokens(3, doc, matches))
        tasks_by_speaker[key] = tasks
        
    return tasks_by_speaker

#     tasks = []

#     for i, match in enumerate(dep_matches):
#         pattern_name = match[0]
#         matches = match[1]

#         if nlp.vocab[pattern_name].text == 'strong_do':

#             tasks.append(doc[matches[0]].text+' ' +
#                          join_dependant_tokens(3, doc, matches))

#     return list(set(tasks))


def get_reminder(text, doc, nlp, dep_matches):

    reminders = []

    for i, match in enumerate(dep_matches):
        pattern_name = match[0]
        matches = match[1]
        if nlp.vocab[pattern_name].text in ['weekday', 'time', 'remind']:
            output = sorted([matches[0], matches[1]])
            reminders.append(["..." + doc[output[0] - 6: output[0]].text, doc[output[0]].text,
                                 doc[output[0]+1:output[1]].text, doc[output[1]].text, doc[output[1]+1: output[1] + 15].text+"..."])
#             reminders.append(join_dependant_tokens(0, doc, matches))

#     return list(set(reminders))
    return reminders


def get_summary(text, doc, nlp, dep_matches):

    return "<summary>"


def get_mbart_ru_summary(text, doc, nlp, dep_matches, lang, model, tokenizer):
    
    discussed = []
    
    for i, match in enumerate(dep_matches):
        pattern_name = match[0]
        matches = match[1]
        if nlp.vocab[pattern_name].text in ['discuss']:
            
            discussed.append(join_dependant_tokens(1, doc, matches))

    input_ids = tokenizer(
    [text],
    max_length=600,
    truncation=True,
    return_tensors="pt",
)["input_ids"]
    
    output_ids = model.generate(
    input_ids=input_ids,
    no_repeat_ngram_size=4
)[0]
    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    big_regex = re.compile('|'.join(map(re.escape, summary_junk)))
    
    if discussed:
        return big_regex.sub(random.choice(discussed_phrases[lang]), random.choice(discussed_phrases[lang])+' '+join_phrases(list(set(discussed)), lang,upper=False) + ' '+summary)
    
    return big_regex.sub(random.choice(discussed_phrases[lang]), summary)


def get_en_summary(text, doc, nlp, dep_matches, lang, model, tokenizer):

    been_done = defaultdict(list)
    strong_been_done = []
    discussed = []
    plans = []
    summary_done = ''
    summary_plan = ''

    for i, match in enumerate(dep_matches):
        pattern_name = match[0]
        matches = match[1]
        if nlp.vocab[pattern_name].text == 'strong_been_done' and len(matches) > 2:

            if doc[matches[0]]._.inflect("VBN"):

                been_done[doc[matches[0]]._.inflect("VBN")].append(doc[matches[2]])

        elif nlp.vocab[pattern_name].text == 'need' and len(matches) > 3:
            
            plans.append(join_dependant_tokens(1, doc, matches))

        elif nlp.vocab[pattern_name].text == 'discuss':

            discussed.append(join_dependant_tokens(1, doc, matches))
    
    for key in been_done.keys():
        subjects = list(set([noun.text for noun in been_done[key]]))
        strong_been_done.append("the "+ join_phrases(subjects, lang, upper=False)[:-1] +' '+['were' if len(subjects)>1 else num_map[lang][been_done[key][0].morph.get("Number")[0]]][0]+" "+key)
        
    if plans:
        summary_plan += '\n'+random.choice(plan_phrases[lang])+' '+join_phrases(plans, lang, upper=False)

    if discussed:
        return random.choice(discussed_phrases[lang])+' ' + join_phrases(discussed, lang, upper=False)+'\n' +join_phrases(strong_been_done, lang)+' '+summary_plan
    else:
        return join_phrases(strong_been_done, lang)+' '+summary_plan


def get_BEEN_DONE(text, doc, nlp, dep_matches):

    extracts_list = []

    for i, match in enumerate(dep_matches):
        pattern_name = match[0]
        matches = match[1]
        if nlp.vocab[pattern_name].text in ['been_done'] and len(matches) > 2:

            output = sorted([matches[0], matches[1], matches[2]])

            extracts_list.append(["..." + doc[output[0] - 6: output[0]].text, doc[output[0]].text, doc[output[0]+1:output[1]].text,
                                 doc[output[1]].text, doc[output[1]+1:output[2]].text, doc[output[2]].text, doc[output[2]+1:output[2]+10].text+"..."])

    return extracts_list


def get_TODO(text, doc, nlp, dep_matches):

    extracts_list = []

    for i, match in enumerate(dep_matches):
        pattern_name = match[0]
        matches = match[1]

        if nlp.vocab[pattern_name].text in ['nsubj_verb_dobj', 'strong_do'] and len(matches) > 3:

            output = sorted([matches[0], matches[1], matches[2], matches[3]])
            extracts_list.append(["..." + doc[output[0] - 6: output[0]].text, doc[output[0]].text, doc[output[0]+1:output[1]].text, doc[output[1]].text,
                                 doc[output[1]+1:output[2]].text, doc[output[2]].text, doc[output[2]+1:output[3]].text, doc[output[3]].text, doc[output[3]+1: output[3] + 15].text+"..."])

        elif nlp.vocab[pattern_name].text in ["can", "today",'want'] and len(matches) > 1:

            output = sorted([matches[0], matches[1]])
            extracts_list.append(["..." + doc[output[0] - 6: output[0]].text, doc[output[0]].text,
                                 doc[output[0]+1:output[1]].text, doc[output[1]].text, doc[output[1]+1: output[1] + 15].text+"..."])

    return extracts_list


def get_keywords(text):

    return []


def get_en_keywords(text):

    keywords = kw_model.extract_keywords(text)[:3]

    return [keyword[0] for i, keyword in enumerate(keywords)]


def extract_dependant_tokens(tokens, dependant_tokens):

    new_dependant_tokens = {}
    for key, value in dependant_tokens.items():
        head_id = key

        for token in tokens:
            if token.head.i == head_id and token.text:
                new_dependant_tokens[token.i] = token.text
    return new_dependant_tokens


def join_dependant_tokens(root_idx, doc, matches):

    tasks = {}
    dependant_tokens = {matches[root_idx]: doc[matches[root_idx]].text}
    tasks[doc[matches[root_idx]].text] = dependant_tokens
    while dependant_tokens:
        dependant_tokens = extract_dependant_tokens(doc, dependant_tokens)
        if dependant_tokens.items() <= tasks[doc[matches[root_idx]].text].items():
            break
        tasks[doc[matches[root_idx]].text].update(dependant_tokens)

    sentence = ' '.join([tasks[doc[matches[root_idx]].text][i]
                        for i in sorted(tasks[doc[matches[root_idx]].text])])

    return sentence


def join_phrases(phrases_list, lang, upper=True):

    sentence = ''
    if len(phrases_list) == 1:
        if upper:
            return phrases_list[0][0].upper() + phrases_list[0][1:]+'.'
        else:
            return phrases_list[0]+"."
    else:
        for i, phrase in enumerate(phrases_list):
            if upper and i == 0:
                sentence += phrase[0].upper() + phrase[1:]+', '

            elif i == len(phrases_list)-1:
                sentence += and_map[lang] +phrase+'.'

            elif i == len(phrases_list)-2:
                sentence += phrase+' '

            else:
                sentence += phrase+', '

        return sentence


num_map = {'ru': {'Plur': '????????', 'Sing': '??????'},
           'en': {'Plur': 'were', 'Sing': 'was'}}

and_map = {'ru':'?? ', "en":"and "}

gen_map = {'Fem': 'femn', 'Masc': 'masc', 'Neut': 'neut'}

plan_phrases = {'ru': ['???????????? ??????????', "???????????? ??????????", "?????????????????? ????????:", "??????????:"],
                'en': ['We now need', "Now we need", "It's time", "Now it's time", "Next is", "Then is"]}

rus_stopwords = ['????', '????????????', "????????????????", '??????????', '????', '????????????', '????????????????', "????????????????", "????????????",
                 '??????????', '??????????????', '????????????????', '??????', "????????", "??????", "????", "????????", "????", "??????"]
pron_stopwords = ["????????????", "??????????????", "??",
                  "????", "????", "??????", "??????", "??????", "??????", "??????"]
verb_stopwords = ['told', 'said', 'had', 'loved', 'see']
noun_stopwords = ['kind', 'microphone', 'screen', 'moment', 'thing']
discussed_phrases = {"en":["We have discussed"], "ru":["??????????????????", "????????????????"]}
summary_junk = ['?????????? ?? ????????', '?? ?????????????????????? ???????????? ?? ???????????????? ?? ??????']

patterns = {
            'need' : {
        
        'ru' : [
            
    {'RIGHT_ID': 'advmod', 'RIGHT_ATTRS': {"LOWER": {"IN": ["??????????","????????","????????????????????", "????????", "??????????", "????????????","????????????????","??????????????????", "??????????????????"]}}},
    {'LEFT_ID': 'advmod', 'REL_OP': '>', 'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'DEP': {"IN":['csubj', "nmod"]},'POS': 'VERB'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'DEP': {"IN":['obj', "conj"]}, 'POS': {"IN":["NOUN", "PRON", "VERB"]}, "LOWER": {"NOT_IN":pron_stopwords}}}
                ],
        
        'en' : [
            
    {'RIGHT_ID': 'mod', 'RIGHT_ATTRS': {'POS': 'VERB',"LOWER": {"IN": ["need","have"]}}},
    {'LEFT_ID': 'mod', 'REL_OP': '>', 'RIGHT_ID': 'x_comp', 'RIGHT_ATTRS': {'DEP': 'xcomp','POS': 'VERB'}},
    {'LEFT_ID': 'x_comp', 'REL_OP': '>', 'RIGHT_ID': 'aux', 'RIGHT_ATTRS': {'DEP': 'aux','POS': 'PART'}},
    {'LEFT_ID': 'x_comp', 'REL_OP': '>', 'RIGHT_ID': 'c_comp', 'RIGHT_ATTRS': {'DEP': 'ccomp', 'POS': {"IN":["PRON", "NOUN", "VERB"]}, "LOWER": {"NOT_IN":pron_stopwords}}}
                ]
                     },
    'could_you' : {
        
        'ru' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {"LOWER": {"IN": ["????????????","??????????","??????","??????????", "????????????"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'part', 'RIGHT_ATTRS': {'DEP': 'advmod',"LOWER": {"IN": ["????", "????"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'x_comp', 'RIGHT_ATTRS': {'DEP': 'xcomp','POS': 'VERB'}}
                ],
        
        'en' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {"LOWER": {"IN": ["could"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj', 'POS': {"IN":["PRON"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'x_comp', 'RIGHT_ATTRS': {'DEP': 'xcomp','POS': 'VERB'}}
                ]
                     },
    'appeared' : {
        
        'ru' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {"LOWER": {"IN": ["????????????????", "??????????????????", "??????????????????"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj', 'POS': {"IN":["NOUN"], "NOT_IN":["PROPN"]}, "LOWER": {"NOT_IN":rus_stopwords}}}
                ],
        
        'en' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {"LOWER": {"IN": ["appeared"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj', 'POS': {"IN":["NOUN"]}}}
                ]
                     },
    'added' : {
        
        'ru' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {"LOWER": {"IN": ["??????????????", "????????????????", "????????????????"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'DEP': {"IN":['obj','pobj', 'dobj']}, 'POS': {"IN":["NOUN"], "NOT_IN":["PROPN"]}, "LOWER": {"NOT_IN":rus_stopwords}}}
                ],
        
        'en' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {"LOWER": {"IN": ["added"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'DEP': {"IN":['obj','pobj', 'dobj']}, 'POS': {"IN":["NOUN"], "NOT_IN":["PROPN"]}}}
                ]
                     },
            'can' : {
        
        'ru' : [
            
    {'RIGHT_ID': 'can', 'RIGHT_ATTRS': {"LOWER": {"IN": ["????????????", "????????????", "??????????????", "??????????????"]}}},
    {'LEFT_ID': 'can', 'REL_OP': '>', 'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'DEP': 'xcomp','POS': 'VERB'}}
                ],
        'en': [
            
    {'RIGHT_ID': 'mod', 'RIGHT_ATTRS': {"LOWER": {"IN": ["can"]}}},
    {'LEFT_ID': 'mod', 'REL_OP': '>', 'RIGHT_ID': 'x_comp', 'RIGHT_ATTRS': {'DEP': 'xcomp','POS': 'VERB'}},
    {'LEFT_ID': 'x_comp', 'REL_OP': '>', 'RIGHT_ID': 'c_comp', 'RIGHT_ATTRS': {'DEP': 'ccomp', 'POS': {"IN":["PRON", "NOUN", "VERB"]}}}
                ]
                     },
    
            'want' : {
        
        'ru' : [
            
    {'RIGHT_ID': 'want', 'RIGHT_ATTRS': {"LOWER": {"IN": ["??????????", "????????","??????????","????????"]}}},
    {'LEFT_ID': 'want', 'REL_OP': '>', 'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'DEP': 'xcomp','POS': 'VERB'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'DEP': 'obj', 'POS': {"IN":["NOUN", "VERB"]}}}
                ],
        'en': [
            
    {'RIGHT_ID': 'mod', 'RIGHT_ATTRS': {'POS': 'VERB',"LOWER": {"IN": ["want", "thinking","planning", "plan","going"]}}},
    {'LEFT_ID': 'mod', 'REL_OP': '>', 'RIGHT_ID': 'x_comp', 'RIGHT_ATTRS': {'DEP': 'xcomp','POS': 'VERB'}},
    {'LEFT_ID': 'x_comp', 'REL_OP': '>', 'RIGHT_ID': 'aux', 'RIGHT_ATTRS': {'DEP': 'aux','POS': 'PART'}},
    {'LEFT_ID': 'x_comp', 'REL_OP': '>', 'RIGHT_ID': 'c_comp', 'RIGHT_ATTRS': {'DEP': 'ccomp', 'POS': {"IN":["PRON", "NOUN", "VERB"]}}}
                ]
                     },
    
            'task' : {
        
        'ru' : [
            
    {'RIGHT_ID': 'noun', 'RIGHT_ATTRS': {'POS': 'NOUN',"LOWER": {"IN": ["????????????", "????????????"]}}},
    {'LEFT_ID': 'noun', 'REL_OP': '>', 'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'DEP': 'csubj','POS': 'VERB'}}
                ],
        'en' : [[
            
    {'RIGHT_ID': 'mod', 'RIGHT_ATTRS': {'POS': 'VERB',"LOWER": {"IN": ["task"]}}},
    {'LEFT_ID': 'mod', 'REL_OP': '>', 'RIGHT_ID': 'x_comp', 'RIGHT_ATTRS': {'DEP': 'xcomp','POS': 'VERB'}},
    {'LEFT_ID': 'x_comp', 'REL_OP': '>', 'RIGHT_ID': 'aux', 'RIGHT_ATTRS': {'DEP': 'aux','POS': 'PART'}},
    {'LEFT_ID': 'x_comp', 'REL_OP': '>', 'RIGHT_ID': 'c_comp', 'RIGHT_ATTRS': {'DEP': 'ccomp', 'POS': {"IN":["PRON", "NOUN", "VERB"]}, "LOWER": {"NOT_IN":pron_stopwords}}}
                ],
            
    [{'RIGHT_ID': 'aux', 'RIGHT_ATTRS': {"LOWER": {"IN": ["is"]}}},
    {'LEFT_ID': 'aux', 'REL_OP': '>', 'RIGHT_ID': 'task', 'RIGHT_ATTRS': {'DEP': 'nsubj',"LOWER": {"IN": ["task","plan"]}}},
    {'LEFT_ID': 'aux', 'REL_OP': '>', 'RIGHT_ID': 'x_comp', 'RIGHT_ATTRS': {'DEP': 'xcomp','POS': 'VERB'}},
    {'LEFT_ID': 'x_comp', 'REL_OP': '>', 'RIGHT_ID': 'part', 'RIGHT_ATTRS': {'DEP': 'aux','POS': 'PART'}}
                ]]
                    },
                
            'discuss' : {
        
        'en' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB',"LOWER": {"IN": ["discuss","discussing"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'obj', 'RIGHT_ATTRS': {'DEP': {"IN" : ['xcomp','dobj']},'POS': {"IN":['VERB',"NOUN"]},"LOWER":{"NOT_IN":noun_stopwords+verb_stopwords}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'aux', 'RIGHT_ATTRS': {'DEP': 'aux','POS': {"IN": ['PART', "AUX"]}}},
                  ],
        
        'ru' : [
            
    {'RIGHT_ID': 'discuss', 'RIGHT_ATTRS': {"LOWER": {"IN": ["??????????????", "??????????????????"]}}},
    {'LEFT_ID': 'discuss', 'REL_OP': '>', 'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'DEP': 'obj', 'POS': {"IN":["NOUN", "VERB"]}}}
                ]
                    },
    
            'remind' : {
        
        'ru' : [
            
    {'RIGHT_ID': 'noun', 'RIGHT_ATTRS': {'POS': 'NOUN',"LOWER": {"IN": ["??????????????????","??????????????????????"]}}},
    {'LEFT_ID': 'noun', 'REL_OP': '>', 'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'DEP': 'csubj','POS': 'VERB'}}
                ],
        
        'en' : [
            
    {'RIGHT_ID': 'remind', 'RIGHT_ATTRS': {'POS': 'VERB',"LOWER": {"IN": ["remind", "forget","reminder", "reminding"]}}},
    {'LEFT_ID': 'remind', 'REL_OP': '>', 'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'DEP': {"IN" : ['ccomp','dobj']},'POS': {"IN":['VERB',"NOUN"]},"LOWER":{"NOT_IN":verb_stopwords+noun_stopwords}}}
                  ]
                },
    
#             'date' : ,
    
            'weekday' : {
        
        'ru' : [
            
            [
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB','MORPH': {'INTERSECTS': ['Tense=Fut','Number=Plur','Tense=Pres',"VerbForm=Inf"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'weekday', 'RIGHT_ATTRS': {'DEP': {"IN": ["nsubj",'obl',"advmod"]},"LOWER": {"IN": ["????????????", "??????????????????????", "??????????????????????","??????????????","??????????","??????????????","??????????????","??????????????","??????????????????????"]}}}
            ],
            
            [
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'LOWER': {'IN': ['????????????????????','??????????????????????']}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'weekday', 'RIGHT_ATTRS': {'DEP': 'obl',"LOWER": {"IN": ["??????????????","????????????", "??????????????????????", "??????????????????????","??????????????","??????????","??????????????","??????????????","??????????????","??????????????????????"]}}}
            ]
                ],
        'en' : 
            [
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'prep', 'RIGHT_ATTRS': {'POS': 'ADP'}},
    {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'weekday', 'RIGHT_ATTRS': {'DEP': 'pobj',"LOWER": {"IN": ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]}}}
            ]},
            
            'time' : {
        
        'ru' : [
            
            [
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>>', 'RIGHT_ID': 'hours', 'RIGHT_ATTRS': {"LOWER" : {"IN" : ['??????', "2", "3", "4", "5","6","7","8","9","10","11", "12", "30", "40","15","10","20","50"]}}},
    {'LEFT_ID': 'hours', 'REL_OP': '>', 'RIGHT_ID': 'minutes', 'RIGHT_ATTRS': {'DEP': 'nummod', "LOWER" : {"IN" : ['??????', "2", "3", "4", "5","6","7","8","9","10","11", "12", "30", "40","15","10","20","50"]}}}
            ],
            
            [
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'LOWER': {'IN': ['????????????????????']}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>>', 'RIGHT_ID': 'hours', 'RIGHT_ATTRS': {'POS': 'NUM'}},
    {'LEFT_ID': 'hours', 'REL_OP': '>', 'RIGHT_ID': 'minutes', 'RIGHT_ATTRS': {'DEP': 'nummod', 'POS': 'NUM'}}
            ]
                ],
        
        'en' : [
                
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>>', 'RIGHT_ID': 'hours', 'RIGHT_ATTRS': {'POS': 'NUM'}},
    {'LEFT_ID': 'hours', 'REL_OP': '>', 'RIGHT_ID': 'minutes', 'RIGHT_ATTRS': {'DEP': 'nummod', 'POS': 'NUM'}}
            ]},
    
            'nsubj_verb_dobj' : { 
        
        'ru' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB',"LOWER":{"NOT_IN":rus_stopwords}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'aux', 'RIGHT_ATTRS': {'DEP': 'aux'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj','POS': {"IN":['PRON',"PROPN"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'DEP': 'obj', 'POS': {"NOT_IN":['NOUN',"PROPN"]}}}
                ],
        'en' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB',"LOWER":{"NOT_IN":verb_stopwords}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'aux', 'RIGHT_ATTRS': {'DEP': 'aux','MORPH': 'VerbForm=Fin'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj','POS': {"IN":['PRON',"PROPN"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'd_object', 'RIGHT_ATTRS': {'DEP': 'dobj','POS': {"NOT_IN":['NOUN',"PROPN"]},"LOWER":{"NOT_IN":noun_stopwords}}}
                ]},
    
            'strong_do' : {
        'ru' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB',"LOWER":{"NOT_IN":rus_stopwords}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'aux', 'RIGHT_ATTRS': {'DEP': 'aux'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj','POS': {"IN":['PRON',"PROPN"]},'LOWER':{"NOT_IN" : ["??????"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'DEP': 'obj', 'POS': {"IN":['NOUN',"PROPN"]},"LOWER":{"NOT_IN":rus_stopwords}}}
                ],
        'en' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB',"LOWER":{"NOT_IN":verb_stopwords}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'aux', 'RIGHT_ATTRS': {'DEP': 'aux','MORPH': 'VerbForm=Fin'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj','POS': {"IN":['PRON',"PROPN"]}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'd_object', 'RIGHT_ATTRS': {'DEP': 'dobj','POS': {"IN":['NOUN',"PROPN"]},"LOWER":{"NOT_IN":noun_stopwords}}}
                ]},
    
            'been_done': {
        
        'ru' : [
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB','MORPH': {'IS_SUPERSET': ['Tense=Past']}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj','POS': 'PRON'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'DEP': 'obj','POS': {"IN":["NOUN"]}, "LOWER": {"NOT_IN":pron_stopwords}}}
                ],
        'en' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB','MORPH': {'IS_SUPERSET': ['Tense=Past']}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj','POS': 'PRON'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'd_object', 'RIGHT_ATTRS': {'DEP': 'dobj',"LOWER":{"NOT_IN":noun_stopwords}}}
                ]},
        'strong_been_done': {
        
        'ru' : [
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB','MORPH': {'IS_SUPERSET': ['Tense=Past']}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj','POS': 'PRON'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'DEP': 'obj','POS': {"IN":["NOUN"]}, "LOWER": {"NOT_IN":pron_stopwords}}}
                ],
        'en' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB', "ORTH":{"NOT_IN": verb_stopwords},'MORPH': {'IS_SUPERSET': ['Tense=Past']}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj','POS':{"IN" : ['PRON', 'PROPN']}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'd_object', 'RIGHT_ATTRS': {'DEP': 'dobj','POS': 'NOUN',"LOWER":{"NOT_IN":noun_stopwords}}}
                ]},
    'today': {
        
        'ru' : [
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB','MORPH': {'IS_SUPERSET': ['Tense=Pres']}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj','POS': 'PRON'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'today', 'RIGHT_ATTRS': {'DEP': 'advmod','LOWER': {"IN":["??????????????"]}}}
                ],
        'en' : [
            
    {'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB','MORPH': {'IS_SUBSET': ['Tense=Pres', 'Tense=Fut']}}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': 'nsubj','POS': 'PRON'}},
    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'today', 'RIGHT_ATTRS': {'DEP': 'advmod',"LOWER":{"IN":['today']}}}
                ]}
    
}
