#from pythainlp.tag import pos_tag
#from pythainlp.tokenize import word_tokenize
import nltk
import spacy
from nltk.tag import tnt
from nltk.corpus import indian
from nltk.tokenize import  word_tokenize
from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForTokenClassification,TokenClassificationPipeline
nltk.download('punkt_tab')

pos_mapper_dict = {"AFX":"ADJ", "CC":"CCONJ", "CD": "NUM", "DT":"DET", "EX":"PRON", \
               "FW":"X", "HYPH":"PUNCT", "IN":"ADP", "JJ": "ADJ", "JJR":"ADJ",\
                "JJS":"ADJ", "LS":"X", "MD":"VERB", "NIL":"X", "NN":"NOUN", "NNO": "NOUN","NNP":"PROPN",\
                "NNS":"NOUN", "PDT":"DET", "POS": "PART", "PRP": "PRON", "PRP$": "DET", "PPO":"ADP",\
                "RB":"ADV", "RBR":"ADV", "RBS":"ADV", "RP":"ADP", "SYM":"SYM","TO": "PART",\
                "UH": "INTJ", "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB",\
                "VBP": "VERB", "VBZ": "VERB", "WDT": "DET", "WP": "PRON","WP$": "DET",\
                "WRB":"ADV", "VM":"VERB", "NST":"NOUN", "INTF":"ADV", "QF":"DET", "PSP":"ADP",\
                 "VAUX":"AUX","ART": "DET", "CSN":"NOUN", "VBI":"VERB","PRR":"PRON", "PAR": "PART",\
                  "ADK":"ADJ", "VBT":"VERB","ADK":"ADJ",  "PRI":"PRON", "NEG":"PART", "CCN":"CCONJ"}


class Hindi_POS:
    def __init__(self):
        self.train_data = indian.tagged_sents("hindi.pos")
        self.hi_pos_tagger = tnt.TnT()
        self.hi_pos_tagger.train(self.train_data)
    def pos_hi(self,sent):
        tokens = word_tokenize(sent)
        tagged_words = self.hi_pos_tagger.tag(tokens)
        tagged_words = [(tup[0],pos_mapper_dict.get(tup[1],tup[1])) for tup in tagged_words]
        return tagged_words    

class Bengali_POS:
    def __init__(self):
        self.train_data = indian.tagged_sents("bangla.pos")
        self.bn_pos_tagger = tnt.TnT()
        self.bn_pos_tagger.train(self.train_data)

    def pos_bn(self,sent):
        tokens = word_tokenize(sent)
        tagged_words = (self.bn_pos_tagger.tag(tokens))
        tagged_words = [(tup[0],pos_mapper_dict.get(tup[1],tup[1])) for tup in tagged_words]
        return tagged_words

class Korean_POS:
    def __init__(self):
    
        tokenizer=AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-base-korean-upos")
        model=AutoModelForTokenClassification.from_pretrained("KoichiYasuoka/roberta-base-korean-upos")
        #pipeline=TokenClassificationPipeline(tokenizer=tokenizer,model=model,aggregation_strategy="simple")
        pipeline=TokenClassificationPipeline(tokenizer=tokenizer,model=model)
        #pipeline=TokenClassificationPipeline(tokenizer=tokenizer,model=model)
        #self.nlp_ko=lambda x:[(x[t["start"]:t["end"]],t["entity_group"]) for t in pipeline(x)]
        self.nlp_ko=lambda x:[(x[t["start"]:t["end"]],t["entity"].replace("B-","").replace("I-","").replace("O-","")) for t in pipeline(x)]
    def pos_ko(self,sent):
        tagged_words = self.nlp_ko(sent)
        tagged_words = [(tup[0],pos_mapper_dict.get(tup[1],tup[1])) for tup in tagged_words]
        return tagged_words

class Thai_POS:
    def __init__self(self):
        pass
    def pos_th(self,sent):
        tokens = word_tokenize(sent)
        word_engine = 'longest'
        pos_engine = 'tltk'
        pos_corpus = 'orchid_ud'
        #[('แมว', 'NOUN'), ('กิน', 'VERB'), ('ปลา', 'NOUN'), (' ', ''), ('แมว', 'NOUN'), ('กิน', 'VERB'), ('ปลา', 'NOUN'), (' ', ''), ('แมว', 'NOUN'), ('กิน', 'VERB'), ('ปลา', 'NOUN')]
        _pos = pos_tag(tokens, corpus=str(pos_corpus), engine=str(pos_engine))
        return _pos
    
class Indo_POS:
    def __init__(self):
        pretrained_name = "w11wo/indonesian-roberta-base-posp-tagger"
        #self.nlp_id = pipeline("token-classification",model=pretrained_name,tokenizer=pretrained_name)
        self.nlp_id = pipeline("ner",model=pretrained_name,tokenizer=pretrained_name)
        #self.nlp_id = pipeline("token-classification", 
                      #model="cahya/bert-base-indonesian-1.5G-finetuned-pos", 
                      #tokenizer="cahya/bert-base-indonesian-1.5G-finetuned-pos")

    def pos_id(self,sent):
        tokens = word_tokenize(sent)
        tagged_words = self.nlp_id(sent)
        #print(pos_tags)
        tagged_words = [(sent[tag['start']:tag["end"]],pos_mapper_dict.get(tag['entity'].replace("B-",""),tag['entity'].replace("B-",""))) for tag in tagged_words]
        tagged_words = [(tup[0],pos_mapper_dict.get(tup[1],tup[1])) for tup in tagged_words]
        return tagged_words   

class Port_POS:
    def __init__(self):
        self.nlp_pt = spacy.load('pt_core_news_md')
    def pos_pt(self,sent):
        document = self.nlp_pt(sent)
        tagged_words = [(token,token.pos_) for token in document]
        return tagged_words


class Germ_POS:
    def __init__(self):
        self.nlp = spacy.load('de_core_news_sm')

    def pos_de(self,sent):
        doc = self.nlp(sent)
        tagged_words = [(t.orth_, t.pos_) for t in doc]
        return tagged_words
    

class French_POS:
    def __init__(self):
        self.nlp = spacy.load('fr_core_news_sm')

    def pos_fr(self,sent):
        doc = self.nlp(sent)
        tagged_words = [(t.orth_, t.pos_) for t in doc]
        return tagged_words

class Span_POS:
    def __init__(self):
        self.nlp = spacy.load('es_core_news_sm')

    def pos_es(self,sent):
        doc = self.nlp(sent)
        tagged_words = [(t.orth_, t.pos_) for t in doc]
        return tagged_words
         
class English_POS:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def pos_en(self,sent):
        doc = self.nlp(sent)
        tagged_words = [(t.orth_, t.pos_) for t in doc]
        return tagged_words
    
import spacy

# --- 2. Chinese (Simplified) ---
class Chinese_POS:
    def __init__(self):
        try:
            self.nlp = spacy.load('zh_core_web_sm')
        except OSError:
            print("Error: Model not found. Run: python -m spacy download zh_core_web_sm")

    def pos_zh(self, sent):
        doc = self.nlp(sent)
        return [(t.orth_, t.pos_) for t in doc]


# --- 3. Russian ---
class Russian_POS:
    def __init__(self):
        try:
            self.nlp = spacy.load('ru_core_news_sm')
        except OSError:
             print("Error: Model not found. Run: python -m spacy download ru_core_news_sm")

    def pos_ru(self, sent):
        doc = self.nlp(sent)
        return [(t.orth_, t.pos_) for t in doc]


# --- 4. Arabic ---
import torch
import functools

# --- PATCH START: Fix for PyTorch 2.6+ / Stanza Compatibility ---
# PyTorch 2.6 made weights_only=True the default, which breaks Stanza models.
# We override the global torch.load to force weights_only=False.

_original_load = torch.load

@functools.wraps(_original_load)
def loose_load(*args, **kwargs):
    # If the caller (Stanza) didn't specify weights_only, default it to False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = loose_load

from transformers import AutoTokenizer


class Arabic_POS:
    def __init__(self):
        print("Loading Arabic Stanza model...")
        import spacy_stanza
        # This will now succeed because torch.load is patched
        self.nlp = spacy_stanza.load_pipeline("ar")

    def pos_ar(self, sent):
        doc = self.nlp(sent)
        return [(t.text, t.pos_) for t in doc]
#training German POS tagger using NLTK

'''
from transformers import pipeline

pretrained_name = "w11wo/indonesian-roberta-base-posp-tagger"

nlp_id = pipeline(
    "token-classification",
    model=pretrained_name,
    tokenizer=pretrained_name
)

from konlpy.tag import Kkma
kkma = Kkma()

from transformers import AutoTokenizer,AutoModelForTokenClassification,TokenClassificationPipeline
tokenizer=AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-base-korean-upos")
model=AutoModelForTokenClassification.from_pretrained("KoichiYasuoka/roberta-base-korean-upos")
pipeline=TokenClassificationPipeline(tokenizer=tokenizer,model=model,aggregation_strategy="simple")
nlp_ko=lambda x:[(x[t["start"]:t["end"]],t["entity_group"]) for t in pipeline(x)]

import spacy
nlp_pt = spacy.load('pt_core_news_md')



def pos_hi(sent):
    tokens = word_tokenize(sent)
    tagged_words = (hi_pos_tagger.tag(tokens))
    tagged_words = [(tup[0],pos_mapper_dict.get(tup[1],tup[1])) for tup in tagged_words]
    return tagged_words


def pos_id(sent):
    tokens = word_tokenize(sent)
    tagged_words = nlp_id(sent)
    #print(pos_tags)
    tagged_words = [(sent[tag['start']:tag["end"]],pos_mapper_dict.get(tag['entity'].replace("B-",""),tag['entity'].replace("B-",""))) for tag in tagged_words]
    tagged_words = [(tup[0],pos_mapper_dict.get(tup[1],tup[1])) for tup in tagged_words]
    return tagged_words

def pos_ko1(sent):
    tagged_words = kkma.pos(sent)
    tagged_words = [(tup[0],pos_mapper_dict.get(tup[1],tup[1])) for tup in tagged_words]
    return tagged_words

def pos_ko(sent):
    tagged_words = nlp_ko(sent)
    return tagged_words

def pos_pt(sent):
    document = nlp_pt(sent)
    tagged_words = [(token,token.pos_) for token in document]
    return tagged_words
        
'''
   



