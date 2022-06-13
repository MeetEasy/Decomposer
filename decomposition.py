from utils import process_json, get_keywords, get_summary, get_tasks, get_reminder, get_BEEN_DONE, get_TODO, patterns
import spacy
from spacy.matcher import Matcher, DependencyMatcher
from langdetect import detect

model_matcher = {'ru' : 'ru_core_news_sm',
                'en' : 'en_core_web_sm'}

def decompose(transcript_json):
    
    text = process_json(transcript_json)
    lang = detect(text)
    
    nlp = spacy.load(model_matcher[lang])
    doc = nlp(text)
    dep_matcher = DependencyMatcher(vocab=nlp.vocab)
    
    for pattern_name in patterns.keys():
        if type(patterns[pattern_name][lang][0]) == list:
            dep_matcher.add(pattern_name, patterns=patterns[pattern_name][lang])
        else: 
            dep_matcher.add(pattern_name, patterns=[patterns[pattern_name][lang]])
            
    dep_matches = dep_matcher(doc)
    
    
    transcript_json['topic'] = get_keywords(text)
    transcript_json['summary'] = get_summary(text, doc, nlp, dep_matches)
    transcript_json['task'] = get_tasks(text, doc, nlp, dep_matches)
    transcript_json['reminder'] = get_reminder(text, doc, nlp, dep_matches)
    
    transcript_json['colored'] = {"BEEN DONE" : get_BEEN_DONE(text, doc, nlp, dep_matches),
                                 "TODO" : get_TODO(text, doc, nlp, dep_matches)}
    
    return transcript_json

