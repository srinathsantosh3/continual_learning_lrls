import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import   BertTokenizer, BertForSequenceClassification, AutoTokenizer
from adapters import AutoAdapterModel, AdapterConfig
from sklearn.preprocessing import LabelEncoder
import sys
from collections import namedtuple, defaultdict
import random
import numpy as np
import json
from torch import nn
from torch.optim.lr_scheduler import StepLR
import argparse
import os
sys.path.append("/home/epsilon/Workbenches/ML_VLM/continual_training/xMDETR/scripts/continual")
from multi_pos import Hindi_POS, Thai_POS,Germ_POS,French_POS, English_POS, Span_POS, Bengali_POS,Korean_POS
from adapters.composition import Stack
from collections import namedtuple, defaultdict
from nltk import word_tokenize
import traceback
import nltk # Added NLTK import
from peft import get_peft_model, LoraConfig, TaskType # Related to Semanntic code switching
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    DataCollatorWithPadding, 
    get_linear_schedule_with_warmup 
)
# Related to Semanntic code switching
seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Semantic Code Switching Variables 
SCORER_MODEL_TYPE = 'Normal' #Normal, Quality, Size  #1000, 5000, 10000, 20000, 30000 #TODO
SCORER_MODEL_TYPE_PARAM = {'Normal': 'ALL', 'Quality':'LOW', 'Size':5000} #TODO
DEVICE_ID = '1' #TODO
SCORER_MODEL = "sentence-transformers/LaBSE"
BASE_MODEL = "sentence-transformers/LaBSE"
THRESHOLD = 0.65
TEMP = 0.5
#Semantic Code Switching Variables 

class ScorePredictorModel(nn.Module):
    def __init__(self, model_name, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name, config=config)
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.base_model = get_peft_model(self.base_model, lora_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.score_regressor = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        
        # FIXED: Return the Tensor directly so torch.sigmoid() works in batch_code_switch
        score_logits = self.score_regressor(cls_output).squeeze(-1)
        return score_logits
    


#pos_hi = Hindi_POS().pos_hi
#pos_th = Thai_POS().pos_th
pos_de = Germ_POS().pos_de
pos_fr = French_POS().pos_fr
pos_en = English_POS().pos_en
#pos_es = Span_POS().pos_es
#pos_bn = Bengali_POS().pos_bn
#pos_ko = Korean_POS().pos_ko
def code_switch_multilingual_pos(s,current_lang, code_switch_dict,POS_TAG,CODESWITCH_TYPE,args,scorers, device, prev_lang):
    #s = s.strip('?')
    tags = eval("pos_"+current_lang)(s)
    print(f"before replacement: {tags}")
    s = [str(tag[0]) for tag in tags]
    #args.code_switch_ratio
    #args.code_switch_ratio = 0.75
    codeswitch_target_count = max(1,int(args.code_switch_ratio*len(s)))
    print(f"words to be replaced:{codeswitch_target_count}")
    words_replacement_pos = 0
    idxs_replace = []
    scorer = scorers[prev_lang]
    
    try:
        if CODESWITCH_TYPE == 'semantic':
            s = ' '.join(s)
            print(f'Semantic: s:{s}')
            tokens = word_tokenize(s)
            metadata = []
            all_candidates = []
            src_candidates = []
            swappable = [i for i, w in enumerate(tokens) if w.lower() in code_switch_dict]
            if not swappable:
                metadata.append(None); 
            
            k = min(max(1, int(args.code_switch_ratio * len(tokens))), len(swappable))
            cands = []
            for idx in swappable:
                trans = code_switch_dict[tokens[idx].lower()][0]
                temp = list(tokens); temp[idx] = trans
                sent = ' '.join(temp)
                all_candidates.append(sent)
                src_candidates.append(s)
                cands.append({"idx": idx, "trans": trans, "sent": sent})
            metadata.append({"tokens": tokens, "k": k, "cands": cands})
            if not all_candidates: return s
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            inputs = tokenizer(all_candidates, return_tensors="pt", padding=True, truncation=True).to(device)
            srcs = tokenizer(s, return_tensors="pt", padding=True, truncation=True).to(device)
            scores = torch.sigmoid(scorer(inputs.input_ids, inputs.attention_mask)).cpu().tolist()
            final_texts, ptr = [], 0
            for m in metadata:
                if m is None: final_texts.append(""); continue
                
                valid = []
                for c in m["cands"]:
                    print(f"Filtering:: {all_candidates[ptr]}:{scores[ptr]}")
                    c["score"] = scores[ptr]; ptr += 1
                    if c["score"] >= THRESHOLD: valid.append(c)
                
                if not valid: 
                    final_texts.append(' '.join(m["tokens"])); continue
                #print(f"Length of valid:{len(valid)}, length of swappable: {len(swappable)}")    
                scores_arr = np.array([v["score"] for v in valid])
                probs = np.exp(scores_arr / TEMP) / np.sum(np.exp(scores_arr / TEMP))
                chosen = np.random.choice(len(valid), size=min(m["k"], len(valid)), replace=False, p=probs)
                res_tokens = m["tokens"]
                ##print(chosen)
                for c_idx in chosen:
                    res_tokens[valid[c_idx]["idx"]] = valid[c_idx]["trans"]
                final_texts.append(' '.join(res_tokens))
                    
            #print("*"*100)
        
            print(final_texts[0]) #returning onlty the string and not the list
            return final_texts[0]
        
    except:
        print(f'In semantic CS exception, here is the input: {s}')
        import traceback
        traceback_string = traceback.print_exc()
        traceback.print_exc()
        return s

    try:
        if CODESWITCH_TYPE == 'random':
            for idx in random.sample(list(range(len(tags))), k=codeswitch_target_count):
                target = s[idx].lower()
                if (words_replacement_pos < codeswitch_target_count) and (target in code_switch_dict):
                        s[idx] = code_switch_dict.get(target)[0]
                        words_replacement_pos+=1                                       

                else:
                    continue

            s = ' '.join(s)
            print(f"random_replacement:{s}")
            return s        


    except:        
        print("In exception:")
        s = ' '.join(s)
        return s 
 
    try:
        #POS Based+ Random word replacement
        for idx,tag in enumerate(tags): #tag[0]: word, tag[1]: POS tag

            if "_" in POS_TAG:
                POS_TAGS = POS_TAG.split("_")
                POS_MATCH_CONDITION = (tag[1] in POS_TAGS)     
            else:
                POS_MATCH_CONDITION = (tag[1] == POS_TAG)


            if (POS_MATCH_CONDITION) & (words_replacement_pos < codeswitch_target_count):
                #target = s[idx].strip(",").strip("'").replace("'s","").lower()
                target = s[idx] 
                idxs_replace += [idx]
                if target in code_switch_dict:
                    s[idx] = code_switch_dict.get(target)[0]
                    words_replacement_pos+=1 
                else:
                    s[idx] = target           
            #print(" Replacement:: "+random.sample(code_switch_dict[target], k=1)[0])
            else:
                continue
        print(f"replacement_after_pos:{s}")


        #Random word replacement
        shuffel_idx = list(range(len(tags)))
        for idx in random.sample(shuffel_idx, k = len(shuffel_idx) ):
            target = s[idx].lower()
            if (words_replacement_pos < codeswitch_target_count) and (target in code_switch_dict) and (idx not in idxs_replace):
                    s[idx] = code_switch_dict.get(target)[0]
                    words_replacement_pos+=1                                       

            else:
                continue     
        print(f"replacement_after_pos+random:{s}")
        
        s = ' '.join(s)+'?'
        return s
    except:
        print("In exception:")
        s = ' '.join(s)+'?'
        return s 

def code_switch_word_drop(s,current_lang,POS_TAG,CODESWITCH_TYPE,args):
    #s = s.strip('?')
    tags = eval("pos_"+current_lang)(s)
    print(f"before replacement: {tags}")
    s = [str(tag[0]) for tag in tags]
    #args.code_switch_ratio
    #args.code_switch_ratio = 0.75
    codeswitch_target_count = max(1,int(args.code_switch_ratio*len(s)))
    print(f"words to be replaced:{codeswitch_target_count}")
    words_replacement_pos = 0
    idxs_replace = []
    try:
        pass
        #Implement smeantic code switching
    except:
        print("In exception:")
        s = ' '.join(s)
        return s 



    try:
        if CODESWITCH_TYPE == 'random':
            for idx in random.sample(list(range(len(tags))), k=codeswitch_target_count):
                if (words_replacement_pos < codeswitch_target_count):
                        #s[idx] = code_switch_dict.get(target)[0]
                        s[idx] = "######" #word drop by setting to empty string
                        words_replacement_pos+=1                                       

                else:
                    continue

            s = ' '.join(s)
            print(f"random_replacement:{s}")
            return s        


    except:        
        print("In exception:")
        s = ' '.join(s)
        return s 
 
    try:
        #POS Based+ Random word replacement
        for idx,tag in enumerate(tags): #tag[0]: word, tag[1]: POS tag

            if "_" in POS_TAG:
                POS_TAGS = POS_TAG.split("_")
                POS_MATCH_CONDITION = (tag[1] in POS_TAGS)     
            else:
                POS_MATCH_CONDITION = (tag[1] == POS_TAG)


            if (POS_MATCH_CONDITION) & (words_replacement_pos < codeswitch_target_count):
                #target = s[idx].strip(",").strip("'").replace("'s","").lower()
             
                idxs_replace += [idx]
                s[idx] = "######"
                words_replacement_pos+=1 
                       
            #print(" Replacement:: "+random.sample(code_switch_dict[target], k=1)[0])
            else:
                continue
        print(f"replacement_after_pos:{s}")


        #Random word replacement
        shuffel_idx = list(range(len(tags)))
        for idx in random.sample(shuffel_idx, k = len(shuffel_idx) ):
            if (words_replacement_pos < codeswitch_target_count)  and (idx not in idxs_replace):
                    #s[idx] = code_switch_dict.get(target)[0]
                    s[idx] = "######"
                    words_replacement_pos+=1                                       

            else:
                continue     
        print(f"replacement_after_pos+random:{s}")

        print(f"before_drop:{s}")
        s = ' '.join(s)+'?'
        s = s.replace("######",'') #removing all the CS words
        print(f"after_drop:{s}") 
        return s
    except:
        print("In exception:")
        s = ' '.join(s)+'?'
        return s 




def get_all_codeswitch(all_langs,CODE_SWITCH_PATH):
    codeswitch_all_lang_dict = {lang1+"-"+lang2:[] for lang1 in all_langs for lang2 in all_langs if lang1!=lang2 }
    if len(all_langs)>1:
        for lang1 in all_langs:
            for lang2 in all_langs:
                if lang1==lang2:
                    continue
                codeswitch_all_lang_dict[lang1+"-"+lang2] = load_code_switch(lang1,lang2,CODE_SWITCH_PATH)[0]    
    return codeswitch_all_lang_dict

def load_code_switch(lang1,lang2,CODE_SWITCH_PATH):
    code_switch_dict = defaultdict(list)
    code_switch_dict_re = defaultdict(list)
    print("cuurent code switch path"+CODE_SWITCH_PATH)
    current_code_switch_path = CODE_SWITCH_PATH+"/"+lang1+"_"+lang2+".tsv"
    with open(current_code_switch_path) as f:
        for line in f:
            k, v = line.split()[0], " ".join(line.split()[1:])
            k,v = k.replace(".",""),v.replace(".","")
            code_switch_dict[k].append(v.lower())
            code_switch_dict_re[v].append(k.lower())
    print("Loading bilingual dictionary:", len(code_switch_dict))
    return code_switch_dict, code_switch_dict_re

class IntentDataset(Dataset):
    def __init__(self, lang,split,splits):
        self.text = []
        self.intent = []
        self.all_intents = []
        self.lang = lang
        self.splits =splits
        split  = "train" if split == "en_cs_sample" else split 
        with open(f"/mnt/MIG_store/Datasets/epsilon/datasets/MTOP/PERLANG/{lang}/{split}.txt") as file:
            
                for line in file.readlines():
                    try:
                        tokens = line.split("\t")
                        if lang == 'th':
                            line_dict = json.loads(tokens[7])
                            words = line_dict['tokens']
                            line = ' '.join(words)
                            self.text.append(line)
                        else:
                            self.text.append(tokens[3])    
                        self.intent.append(tokens[1])
                    except:
                        continue      

        self.all_intents = list(set(self.intent))
        self.intent  = self.get_labels()
        self.max_length = max([len(txt) for txt in self.text])
    def __len__(self):
        return len(self.text)
    
    def get_labels(self):
        le = LabelEncoder()
        all_intents = []
        for split in self.splits:
            if "memory" in split: # create memory using traning data
                    continue
            split  = "train" if split == "en_cs_sample" else split
            with open(f"/mnt/MIG_store/Datasets/epsilon/datasets/MTOP/PERLANG/{self.lang}/{split}.txt") as file:
                
                for line in file.readlines():
                    try:
                        tokens = line.split("\t")    
                        all_intents.append(tokens[1])
                    except:
                        continue      

        le.fit(all_intents)
        target = le.transform(self.intent)
        return target
    
    def __getitem__(self, idx):
        return {
            "text": self.text[idx],
            "labels": self.intent[idx]
        }


def accuracy(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.mean(y_pred == y_true)


def bootstrap_accuracy(y_true, y_pred,batch_size, B=1000):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(B):
        idx = np.random.choice(n, batch_size, replace=True)
        score = accuracy(y_pred[idx], y_true[idx])
        scores.append(score)
    scores = np.array(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    ci_lower = np.percentile(scores, 2.5)
    ci_upper = np.percentile(scores, 97.5)
    return mean_score, std_score, (ci_lower, ci_upper)



def evaluate_modelv1(model,dataloaders, lang,split,TOTAL_INPUTS,task_id,tokenizer,MAX_LENGTH,device,adapters_dict,BATCHSIZE,args):
    if args.only_lang_adapters:
        if lang not in args.train_from_scratch:
            model.set_active_adapters([lang])
        else:
            model.set_active_adapters([adapters_dict[lang]])

    else:    
        if lang not in args.train_from_scratch:
            model.set_active_adapters([lang, "ccl"])
        else:
            model.set_active_adapters([adapters_dict[lang], "ccl"])
    #model.set_active_adapters("lm_adapter_single")
    #model.to("cuda:0")
    model.eval()
    print(f" Evaluation adapter summary: {model.adapter_summary()}")
    accuracy = []
    total = 0
    all_preds = []
    all_labels = []
    for batch in dataloaders[lang][split]:
        print(f"evaluation_language is {lang} and data is :{batch['text']}, task_id:{task_id}")
        encoding = tokenizer.batch_encode_plus( batch["text"], max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt' ) 
        batch['input_ids'] = encoding["input_ids"] 
        batch['attention_mask'] = encoding["attention_mask"] 
        batch['labels'] = torch.tensor(batch["labels"])
        labels = batch['labels']
        del batch['text']
        del batch['labels']
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch['input_ids'],batch['attention_mask'])
        #print(f"size of tensor:{outputs.logits.size()}")
        logits =  outputs.logits    
        predictions = torch.argmax(logits, dim=1)
        predictions = predictions.to('cpu')
        #print(f"len of predictions:{len(predictions)}, len of labels: {len(labels)}")
        accuracy += [((predictions == labels).sum().item())/BATCHSIZE]
        #print(f"current correct predictions: {(predictions == labels).sum().item()}")
        total+=len(predictions)
        #print(f"current_accuracy:{((predictions == labels).sum().item())/256.0}")
        all_preds+=predictions
        all_labels+=labels

    #accuracy /= total
   
    mean_accuracy,sd,_ = bootstrap_accuracy(all_labels,all_preds,BATCHSIZE)
    #mean_accuracy,sd = np.average(accuracy), np.std(accuracy)
    print(f"accuracy:{mean_accuracy}, sd: {sd}")

    return mean_accuracy,sd


def evaluate_model(model,dataloaders, lang,split,TOTAL_INPUTS,task_id,tokenizer,MAX_LENGTH,device,adapters_dict,BATCHSIZE,args):
    if args.only_lang_adapters:
        if lang not in args.train_from_scratch:
            model.set_active_adapters([lang])
        else:
            model.set_active_adapters([adapters_dict[lang]])

    else:    
        if lang not in args.train_from_scratch:
            model.set_active_adapters([lang, "ccl"])
        else:
            model.set_active_adapters([adapters_dict[lang], "ccl"])
    #model.set_active_adapters("lm_adapter_single")
    #model.to("cuda:0")
    model.eval()
    print(f" Evaluation adapter summary: {model.adapter_summary()}")
    accuracy = []
    total = 0
    for batch in dataloaders[lang][split]:
        print(f"evaluation_language is {lang} and data is :{batch['text']}, task_id:{task_id}")
        encoding = tokenizer.batch_encode_plus( batch["text"], max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt' ) 
        batch['input_ids'] = encoding["input_ids"] 
        batch['attention_mask'] = encoding["attention_mask"] 
        batch['labels'] = torch.tensor(batch["labels"])
        labels = batch['labels']
        del batch['text']
        del batch['labels']
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch['input_ids'],batch['attention_mask'])
        #print(f"size of tensor:{outputs.logits.size()}")
        logits =  outputs.logits    
        predictions = torch.argmax(logits, dim=1)
        predictions = predictions.to('cpu')
        #print(f"len of predictions:{len(predictions)}, len of labels: {len(labels)}")
        accuracy += [((predictions == labels).sum().item())/BATCHSIZE]
        #print(f"current correct predictions: {(predictions == labels).sum().item()}")
        total+=len(predictions)
        #print(f"current_accuracy:{((predictions == labels).sum().item())/256.0}")

    #accuracy /= total
    print(f"accuracy:{accuracy}")
    mean_accuracy,sd = np.average(accuracy), np.std(accuracy)
    return mean_accuracy,sd

def set_trainable_parameters(model):
    for param in model.named_parameters():
        if "embeddings" not in param[0] and "lm_head" not in param[0] and "task_lm_adapter" not in param[0] and f"{CURRENT_LANG}_lm_adapter" not in param[0]:
            param[1].requires_grad = False   
        else:
            param[1].requires_grad = True 
    return model

def set_classhead_parameters(model):
    for param in model.named_parameters():
        if "heads.ccl" in param[0]:
            param[1].requires_grad = True
        else: 
            param[1].requires_grad = False

def get_trainable_parameters(model):
    print("Trainable Layers...")
    for param in model.named_parameters():
        
        if param[1].requires_grad:
            print(param[0]) 
    
def load_scorers(device,all_langs):
    
    scorer = ScorePredictorModel(SCORER_MODEL).to(device)
    all_scorers = {lang:scorer for lang in all_langs}
    for TARGET_LANG in all_langs:    
        if SCORER_MODEL_TYPE == 'Normal':
            al_path = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/best_model_en-{TARGET_LANG}_score_prediction.pth"

        elif SCORER_MODEL_TYPE == 'Quality':
            al_path = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/best_model_en-{TARGET_LANG}_score_prediction_ALL_{SCORER_MODEL_TYPE_PARAM['Quality']}.pth"

        elif SCORER_MODEL_TYPE == 'Size':
            al_path = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/best_model_en-{TARGET_LANG}_score_prediction_{SCORER_MODEL_TYPE_PARAM['Size']}_ALL.pth"

        print(f"****Loaded model path: {al_path}")

        if os.path.exists(al_path):
            print(f"[INFO] Loading trained weights from: {al_path}")
            state_dict = torch.load(al_path, map_location=device)
            
            # Robust loading: handles potential key mismatches from LoRA/PEFT wrapping
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("regressor.", "score_regressor.") # Alignment check
                new_state_dict[new_key] = v
                
            #scorer.load_state_dict(new_state_dict, strict=False)
            scorer.load_state_dict(state_dict)
        else:
            print(f"[WARN] Alignment weights not found. Using raw LaBSE.")
        
        all_scorers[TARGET_LANG] = scorer.eval()

    print('Completed loading scorers...')    

    return all_scorers

def train_model(args):
    adapters_dict = {"en": "AdapterHub/bert-base-multilingual-cased-en-wiki_pfeiffer", 
                     "de":"AdapterHub/bert-base-multilingual-cased-de-wiki_pfeiffer",
                     "hi": "AdapterHub/bert-base-multilingual-cased-hi-wiki_pfeiffer",
                     "fr": "AdapterHub/bert-base-multilingual-cased-fr-wiki_pfeiffer",
                     "es": "AdapterHub/bert-base-multilingual-cased-es-wiki_pfeiffer",
                     "th": "custom_adapter_th",
                     "bn": "custom_adapter_bn",
                     "ko": "AdapterHub/bert-base-multilingual-cased-ko-wiki_pfeiffer"
                     }
    
    args.train_from_scratch = "th_bn"

    '''
    
    if args.roberta_adapters:
        adapters_dict = {"en": "AdapterHub/xlm-roberta-base-en-wiki_pfeiffer", 
                        "de":"AdapterHub/xlm-roberta-base-de-wiki_pfeiffer",
                        "hi": "AdapterHub/xlm-roberta-base-hi-wiki_pfeiffer",
                        "fr": "AdapterHub/bert-base-multilingual-cased-fr-wiki_pfeiffer",
                        "es": "AdapterHub/xlm-roberta-base-es-wiki_pfeiffer",
                        "th": "AdapterHub/xlm-roberta-base-th-wiki_pfeiffer",
                        "bn": "custom_adapter_bn",
                        "ko": "AdapterHub/bert-base-multilingual-cased-ko-wiki_pfeiffer"
                        }
        args.train_from_scratch = "bn"
    '''
    if args.roberta_adapters:
        adapters_dict = {"en": "AdapterHub/xlm-roberta-base-en-wiki_pfeiffer", 
                        "de":"AdapterHub/xlm-roberta-base-de-wiki_pfeiffer",
                        "hi": "AdapterHub/xlm-roberta-base-hi-wiki_pfeiffer",
                        "fr": "custom_adapter_fr",
                        "es": "AdapterHub/xlm-roberta-base-es-wiki_pfeiffer",
                        "th": "AdapterHub/xlm-roberta-base-th-wiki_pfeiffer",
                        "bn": "custom_adapter_bn",
                        "ko": "custom_adapter_ko"
                        }
        args.train_from_scratch = "fr_bn_ko"    
    
    all_langs = args.all_langs.split("_")
    #all_langs = ["en", "de", "fr", "es", "hi", "th"]
    splits = ["train", "test", "eval", "memory","memory2","en_cs_sample"]
    #MODEL_TYPE = 'bert-base-multilingual-cased'
    MODEL_TYPE = args.model
    BATCHSIZE = args.batch_size
    CODE_SWITCH_PATH = args.CODE_SWITCH_PATH 
    #"/home/epsilon/Workbenches/ML_VLM/continual_training/MTOP/dicts"
    EPOCHS = args.epochs
    REPLAY_FREQ = args.replay_freq
    #CODESWITCH_REPLAY = args.codeswitch_replay  # False = target langauge sentence replay, True = Code switch replay
    REPLAY_TYPE = args.replay_type
    REPLAY = args.replay # True = replay of previous languages - continual learning. False = no continual learning 
    UPOS_EVAL = args.UPOS_EVAL
    CODESWITCH_TYPE = args.codeswitch_type
    MEMORY_SIZE = int(args.memory_size)
    dataset = IntentDataset("en","train",splits)
    INTENTS = len(dataset.all_intents)
    MAX_LENGTH = dataset.max_length
    DEVICE_ID = str(args.device_id)
    MODEL_PATH = args.model_path
    print(f"MAX_LENGTH: {MAX_LENGTH}")
    dataloader_langs = list(set(all_langs+[args.lang_cs]))
    codeswitch_all_lang_dict = get_all_codeswitch(dataloader_langs,CODE_SWITCH_PATH)
    output_path = MODEL_PATH+"/"+args.all_langs
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    RESULT_FILE = output_path+"/"+args.result_file    
    #for batch_size in [256]:
    JSON_FILE = output_path+"/"+args.exp_name+".json"
    f = open(JSON_FILE, 'w')
    json.dump(vars(args), f, indent=4)
    f.close()
    dataloaders = dict()

    
    
    cs_memory =  {f"{lang1}_{lang2}":[] for lang1 in all_langs for lang2 in all_langs if lang1!=lang2}
    
    dataloaders = {lan:{split:[]} for lan in dataloader_langs for split in splits}
    for lang in dataloader_langs:
        for split in splits:
                TOTAL_INPUTS = len(dataset)
                
                if 'memory' in split: #handle memory dataset
                    train_dataset = IntentDataset(lang,"train",splits)
                    train_dataset_2 = IntentDataset(lang,"train",splits)
                    #MEMORY_SIZE = 750#as per the CCL paper - Table17
                    memory_idxs = random.sample(list(range(len(train_dataset))),MEMORY_SIZE)
                    memory_dataset = Subset(train_dataset,memory_idxs)
                    dataloaders[lang][split] = DataLoader(memory_dataset, batch_size=BATCHSIZE, shuffle=True) 
                        
                         
                else: # Handles rest: of the cases other than memory
                    if split == "en_cs_sample":
                        dataset = IntentDataset(lang,"train",splits)
                    else:
                        dataset = IntentDataset(lang,split,splits)    
                    #To get the same first batch every time, We will eventually wrigte code for bigger memory
                    if split == "train": # resize the training data to 10000
                        TRAIN_SIZE = args.train_size #10000
                        import random
                        memory_idxs = random.sample(list(range(len(dataset))),TRAIN_SIZE)
                        dataset = Subset (dataset,memory_idxs)

                    elif split == "en_cs_sample" and lang == "en": # resize the training data to 10000
                        TRAIN_SIZE = args.train_size #10000
                        import random
                        sample_size = int(args.en_cs_percent*TRAIN_SIZE)
                        memory_idxs = random.sample(list(range(len(dataset))),sample_size)
                        dataset = Subset (dataset,memory_idxs)    

                    dataloaders[lang][split] = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)    
                print(f"lang: {lang}, split {split}, dataset size{TOTAL_INPUTS}, no of batches of dataloader {len(dataloaders[lang][split])}")
    # Load the tokenizer
    
    
    from transformers import AutoConfig

    #tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
    #from transformers import  AdapterConfig
    #config = AdapterConfig.load("pfeiffer", dropout=0.2)

    config = AutoConfig.from_pretrained(MODEL_TYPE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)

    device = "cuda:"+DEVICE_ID
    model = AutoAdapterModel.from_pretrained(MODEL_TYPE, config = config)

    scorers = load_scorers(device,all_langs)

    if args.only_lang_adapters:
         model.add_classification_head("ccl", num_labels=INTENTS)

    elif args.task:
        model.add_adapter("ccl")    
        model.add_classification_head("ccl", num_labels=INTENTS)

    #calculate final performance
    lang_acc =  {lang: 0 for  lang in  all_langs}
    all_accuracy = []    
     
    for lang_idx,current_lang in enumerate(all_langs):
        print(f"Current_training_language:{current_lang}")
        from adapters import  SeqBnConfig
        #optimizer = torch.optim.AdamW(model.classification_heads[lang_idx].parameters(), lr=3e-5)
        loss_val = 0
        #adapter_config = AdapterConfig.load(type="seq_bn")
        adapter_config = SeqBnConfig(reduction_factor=16) 
        adapter_name_str = adapters_dict[current_lang]
        #from transformers.adapters import AdapterConfig
        
            
        if current_lang in args.train_from_scratch:
            
             model.add_adapter(adapters_dict[current_lang],adapter_config)

        else:
            #adapter_config = AdapterConfig.load("pfeiffer")
            model.load_adapter(adapters_dict[current_lang],adapter_config)

        print(f"before declaration adapter summary: {model.adapter_summary()}")
        #this trains only language adapter
        if args.only_lang_adapters:
            if current_lang not in args.train_from_scratch:
                model.train_adapter([current_lang])
                #model.set_active_adapters([current_lang, "ccl"])
            else:
                model.train_adapter([adapters_dict[current_lang]])
                #model.set_active_adapters([adapters_dict[current_lang], "ccl"])

        #this trains both task adapter and language adapter
        elif args.task and not args.cll_train :
            if current_lang not in args.train_from_scratch:
                model.train_adapter([current_lang,"ccl"])
                #model.set_active_adapters([current_lang, "ccl"])
            else:
                model.train_adapter([adapters_dict[current_lang],"ccl"])
                #model.set_active_adapters([adapters_dict[current_lang], "ccl"])
        # this trains only task adapter(cll adapter)
        elif args.cll_train:
            if current_lang in args.train_from_scratch:
                model.train_adapter([adapters_dict[current_lang],"ccl"])
                model.set_active_adapters([adapters_dict[current_lang], "ccl"])
            else:
                model.train_adapter(["ccl"])
                model.set_active_adapters([current_lang, "ccl"])

            
        #fall back option: that trains only language adapter
        else:
            if current_lang not in args.train_from_scratch:
                model.train_adapter([current_lang])
                model.set_active_adapters([current_lang])
            else:
                model.train_adapter([adapters_dict[current_lang]])
                model.set_active_adapters([adapters_dict[current_lang]])

            
        model.to(device)
            
        print(f"post declaration adapter summary: {model.adapter_summary()}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        #scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        import adapters
        adapters.init(model)
        if  lang_idx!=0:
            prev_langs = all_langs[0:lang_idx]
            
        else:
            prev_langs = []

        get_trainable_parameters(model)
           
        best_accuracy = 0   
        prev_lang_acc= {lang: 0 for  lang in  prev_langs}
        
        for epoch in range(EPOCHS):
            model.train()
            batch_idx = 0
             #all_langs = ["hi","th"]
            lang_count = {lang: 0 for  lang in  all_langs}
            
            print(f"train active adapter: {model.active_adapters}, current_lang:{current_lang}")
            get_trainable_parameters(model)
            for batch in dataloaders[current_lang]['train']:
                print(f"batch_idx:{batch_idx}, lang_idx{lang_idx}")


                if ((batch_idx%REPLAY_FREQ)==0) and (batch_idx!=0) and (lang_idx!=0) and REPLAY:
                    print(f"Inside if condition,replay_type: {REPLAY_TYPE}")
                    import numpy as np
                    categories = []
                    probabilities = []
                    if args.acc_boost:
                        for k,v in prev_lang_acc.items():
                            categories+=[k]
                            probabilities+=[1-v] 
                        all_probs = [p/sum(probabilities) for p in probabilities ]   
                        prev_lang = np.random.choice(categories, size=1, p=all_probs)[0]
                    else:    
                        prev_lang_idx =  np.random.choice(len(prev_langs),1)[0]
                        prev_lang = all_langs[prev_lang_idx]
                    ##Important condition: only CLL adapter is trained here and for only language adapter condition only classification head 
                    if args.only_lang_adapters:
                        #pass #train only classification head
                        #get_trainable_parameters(model)
                        #train only classification head
                        set_classhead_parameters(model)
                        print("only language replay condition")
                        #get_trainable_parameters(model)
                    else:    
                        model.train_adapter(["ccl"])

                    # Below are the conditions to activate adapters for feature extraction purposes.
                    # If ccl adapter is trained and both language and cll adapter is set activte it means, in replay
                    # For feature extraction language adapter and cll adapter is used. But only cll adapter gets updated
                    # TODO: for only lang adapters, only classification head has to get udated and not language adapters   
                    if args.only_lang_adapters:
                        
                        get_trainable_parameters(model)

                    elif args.replay_adapter=="clang_ccl" and (not args.lang_cs=="en"):
                        if current_lang not in args.train_from_scratch:
                            model.set_active_adapters([current_lang,"ccl"])
                        else:
                            model.set_active_adapters([adapters_dict[current_lang],"ccl"])
                                 #experiment on previous lang n current lang

                    elif args.replay_adapter=="clang_ccl" and (args.lang_cs=="en"):
                        
                        
                        model.set_active_adapters(["en","ccl"])
                                 
                    elif args.replay_adapter=="plang_clang_ccl":
                        if current_lang in args.train_from_scratch:
                            model.set_active_adapters([adapters_dict[current_lang],prev_lang,"ccl"])
                        elif prev_lang in args.train_from_scratch:
                            model.set_active_adapters([current_lang,adapters_dict[prev_lang],"ccl"])
                        else:
                            model.set_active_adapters([current_lang,prev_lang,"ccl"])        
                    else:
                        model.set_active_adapters(["ccl"])

                    print(f"in replay adapter summary: {model.adapter_summary()}")
                    #model.set_active_adapters("ccl"
                    get_trainable_parameters(model)
                    
                    lang_count[prev_lang]+=1
                    if REPLAY_TYPE == "cs_memory": #selects beteen codeswitch and memory replay
                       import random
                       REPLAY_TYPE = random.choice(["code_switch","memory"])
                       if epoch < int(args.mem_cs_thr):
                           REPLAY_TYPE = "memory"

                       print(f"the choice taken:{REPLAY_TYPE}")

                    if REPLAY_TYPE == "code_switch":
                        if (args.lang_cs=="en") and (prev_lang!="en"):
                            data_iterator = iter(dataloaders["en"]['train'])
                            single_batch = next(data_iterator)    
                            batch['text'] = single_batch["text"]
                            batch['labels'] = single_batch["labels"]
                            #TODO: change current_lang to english to pickup correct POS
                            batch["text"] = [code_switch_multilingual_pos(txt,'en',codeswitch_all_lang_dict["en"+"-"+prev_lang],UPOS_EVAL,CODESWITCH_TYPE,args, scorers,device,prev_lang) for txt in batch["text"]]
                        elif (args.lang_cs=="en") and (prev_lang=="en"):#if previous language is  english, we can codswitch with current and prev language - 
                            data_iterator = iter(dataloaders["en"]['train'])
                            single_batch = next(data_iterator)    
                            batch['text'] = single_batch["text"]
                            batch['labels'] = single_batch["labels"]

                        elif (args.lang_cs!="en"):

                            if args.lang_cs != prev_lang:
                                data_iterator = iter(dataloaders[args.lang_cs]['train'])
                                single_batch = next(data_iterator)    
                                batch['text'] = single_batch["text"]
                                batch['labels'] = single_batch["labels"]
                                #TODO: change current_lang to english to pickup correct POS
                                batch["text"] = [code_switch_multilingual_pos(txt,args.lang_cs,codeswitch_all_lang_dict[args.lang_cs+"-"+prev_lang],UPOS_EVAL,CODESWITCH_TYPE,args) for txt in batch["text"]]    
                            else:
                                data_iterator = iter(dataloaders[args.lang_cs]['train'])
                                single_batch = next(data_iterator)    
                                batch['text'] = single_batch["text"]
                                batch['labels'] = single_batch["labels"]
        

                    elif REPLAY_TYPE == "code_switch_sample":
                        if (args.lang_cs=="en") and (prev_lang!="en"):
                            data_iterator = iter(dataloaders["en"]['en_cs_sample'])
                            single_batch = next(data_iterator)    
                            batch['text'] = single_batch["text"]
                            batch['labels'] = single_batch["labels"]
                            #TODO: change current_lang to english to pickup correct POS
                            batch["text"] = [code_switch_multilingual_pos(txt,'en',codeswitch_all_lang_dict["en"+"-"+prev_lang],UPOS_EVAL,CODESWITCH_TYPE,args) for txt in batch["text"]]

                        elif (args.lang_cs=="en") and (prev_lang=="en"):#if previous language is  english, we can codswitch with current and prev language - 
                            data_iterator = iter(dataloaders["en"]['en_cs_sample'])
                            single_batch = next(data_iterator)    
                            batch['text'] = single_batch["text"]
                            batch['labels'] = single_batch["labels"]    
    
                    elif REPLAY_TYPE == "code_switch_drop":
                        
                        data_iterator = iter(dataloaders[prev_lang]["train"])
                        single_batch = next(data_iterator)    
                        batch['text'] = single_batch["text"]
                        batch['labels'] = single_batch["labels"]
                        #TODO: change current_lang to english to pickup correct POS
                        batch["text"] = [code_switch_word_drop(txt,prev_lang,UPOS_EVAL,CODESWITCH_TYPE,args) for txt in batch["text"]]
                    
                    elif REPLAY_TYPE == "memory":
                        data_iterator = iter(dataloaders[prev_lang]['memory'])
                        single_batch = next(data_iterator)    
                        batch['text'] = single_batch["text"]
                        batch['labels'] = single_batch["labels"]

                    elif REPLAY_TYPE == "inter_cs_memoryv1":
                        data_iterator = iter(dataloaders[prev_lang]['memory'])
                        single_batch = next(data_iterator)    
                        batch['text'] = single_batch["text"]
                        batch['labels'] = single_batch["labels"]
                        batch["text"] = [code_switch_multilingual_pos(txt,prev_lang,codeswitch_all_lang_dict[prev_lang+"-"+current_lang],UPOS_EVAL,CODESWITCH_TYPE,args)if (i%2 == 0) else batch["text"]  for i,txt in enumerate(batch["text"])]

                    elif REPLAY_TYPE == "inter_cs_memoryv2":
                        data_iterator = iter(dataloaders[prev_lang]['memory'])
                        single_batch = next(data_iterator)    
                        batch['text'] = single_batch["text"]
                        batch['labels'] = single_batch["labels"]
                        first_half_batch = [txt for i,txt in enumerate(batch["text"]) if (i%2 == 0)]
                        second_half_batch = [code_switch_multilingual_pos(txt,prev_lang,codeswitch_all_lang_dict[prev_lang+"-"+current_lang],UPOS_EVAL,CODESWITCH_TYPE,args) for i,txt in enumerate(batch["text"]) if (i%2 != 0)]
                        print(f"first batch:{first_half_batch}, type :{type(first_half_batch)}")
                        print(f"second batch:{second_half_batch}, type:{type(second_half_batch)}")
                        mix_batch = first_half_batch+ second_half_batch
                        print(f"mix_bath: {mix_batch}")
                        batch["text"] = mix_batch

                    else:
                        data_iterator = iter(dataloaders[prev_lang]['train'])
                        single_batch = next(data_iterator)    
                        batch['text'] = single_batch["text"]
                        batch['labels'] = single_batch["labels"]
                        
                
                print(f"in training adapter summary: {model.adapter_summary()}")
                REPLAY_TYPE = args.replay_type
                batch_idx+=1
                #encoding = tokenizer.batch_encode_plus( batch["text"], max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt' ) 
                try:
                    encoding = tokenizer.batch_encode_plus( batch["text"], padding='max_length', truncation=True, return_tensors='pt' ) 
                except:
                    print(f"In the exception during tokenization: {batch['text']}")
                batch['input_ids'] = torch.tensor(encoding["input_ids"] ).to(device)
                batch['attention_mask'] = torch.tensor(encoding["attention_mask"] ).to(device)
                batch['labels'] = torch.tensor(batch['labels']).to(device)
    
                print(f"train current langauge {current_lang} and data is {batch['text']}")
                #outputs = model(**batch)
                outputs = model(input_ids=batch['input_ids'], labels=batch['labels'], attention_mask=batch['attention_mask'])
                #outputs = model(batch['input_ids'],batch['attention_mask'],batch["labels"])
                loss  = outputs.loss
                #loss_fct = nn.CrossEntropyLoss()
                #loss = loss_fct(logits, labels = batch['labels'])
                loss.backward()
                optimizer.step()
                #scheduler.step()
                optimizer.zero_grad()
                loss_val +=loss.item()
                #predictions = torch.argmax(logits, dim=-1)
                #print(f"accuracy: {(predictions ==  batch['labels']).sum().item()}, batch_idx:{batch_idx}, lang:{current_lang}")
                ##After replay we are again making adapters trainable
                #we train only language adapters
                if args.only_lang_adapters:
                    if current_lang not in args.train_from_scratch:
                        model.train_adapter([current_lang])
                        model.set_active_adapters([current_lang])
                    else:
                        model.train_adapter([adapters_dict[current_lang]])
                        model.set_active_adapters([adapters_dict[current_lang]])
                    print("after completion of replay...")    
                    get_trainable_parameters(model)    

                elif args.train_lang_adapter:
                    #in case of only CLL train, it trains only cll adapter and adapters that need to be trained  from scratch
                    if args.cll_train:
                        if current_lang not in args.train_from_scratch:
                            model.train_adapter(["ccl"])
                            model.set_active_adapters([current_lang,"ccl"]) 
                        
                        else:
                            model.train_adapter([adapters_dict[current_lang],"ccl"])
                            #model.set_active_adapters([adapters_dict[current_lang],"ccl"])
                    # in this case, train both language and task adapter
                    else:
                        if current_lang not in args.train_from_scratch:
                            model.train_adapter([current_lang, "ccl"])
                        else:
                            model.train_adapter([adapters_dict[current_lang], "ccl"])
                    
            file  = open(RESULT_FILE, 'a')    
            accuracy,sd = evaluate_model(model,dataloaders,current_lang,"test",TOTAL_INPUTS,lang_idx,tokenizer,MAX_LENGTH,device,adapters_dict,BATCHSIZE,args)    
            print(f"Accuracy:{accuracy}, sd:{sd},lang:{current_lang}, epoch:{epoch}, loss:{loss}")
            result = f"{BATCHSIZE},{current_lang},{current_lang},{epoch}, {accuracy}, {sd},  {loss}, {lang_count} \n"
            file.write(result)
            
            if lang_idx == (len(all_langs)-1) and (epoch == (EPOCHS-1)):
                all_accuracy += [accuracy]
                lang_acc[current_lang] =  accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            if len(prev_langs)>0:
                for prev_idx, prev_lang in enumerate(prev_langs):
                    accuracy,sd = evaluate_model(model,dataloaders,prev_lang,"test",TOTAL_INPUTS, prev_idx,tokenizer,MAX_LENGTH,device,adapters_dict,BATCHSIZE,args)
                    result = f"{BATCHSIZE},{current_lang},{prev_lang},{epoch}, {accuracy}, {sd}, {lang_count} \n"
                    prev_lang_acc[prev_lang] = accuracy
                    file.write(result)
                    if lang_idx == (len(all_langs)-1) and (epoch == (EPOCHS-1)):
                        all_accuracy += [accuracy]
                        lang_acc[prev_lang] =  accuracy    
            file.close()
            model.to(device)
            print(f"after evaluation model summary:{model.adapter_summary()}")

            #this will train language adapter only first batch
            if args.only_lang_adapters:
                if current_lang not in args.train_from_scratch:
                    model.train_adapter([current_lang])
                else:
                    model.train_adapter([adapters_dict[current_lang]])

            elif args.cll_train:
                if current_lang not in args.train_from_scratch:
                    model.train_adapter(["ccl"])
                    model.set_active_adapters([current_lang,"ccl"]) 
                    
                else:
                    model.train_adapter([adapters_dict[current_lang],"ccl"])
                    #model.set_active_adapters([adapters_dict[current_lang],"ccl"])

            else:
                if current_lang not in args.train_from_scratch:
                    model.train_adapter([current_lang, "ccl"])
                else:
                    model.train_adapter([adapters_dict[current_lang], "ccl"])
                

            print(f"after epoch model summary:{model.adapter_summary()}")
        if not args.only_lang_adapters:    
            model.save_adapter(f"{output_path}/{args.exp_name}_ccl_{current_lang}","ccl")

    
    print("Deleting the model and dataloaders..")
    import numpy as np
    args.final_perf = np.mean(all_accuracy) 
    args.final_perf_dict = str(lang_acc)
    import time
    ts = time.time()
    import datetime
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    args_dict = vars(args)
    args_dict["time_stamp"] = time_stamp
    f = open(JSON_FILE, 'w')
    json.dump(args_dict, f, indent=4)  
    f.close()
    print("Saving the model and adapters..")
    model.save_pretrained(output_path+"/"+args.exp_name)
    for current_lang in all_langs:
        if current_lang not in args.train_from_scratch:
            model.save_adapter(f"{output_path}/{args.exp_name}_{current_lang}",current_lang)
        else:
            model.save_adapter(f"{output_path}/{args.exp_name}_{adapters_dict[current_lang]}",adapters_dict[current_lang])
    
    if not args.only_lang_adapters:
        model.save_adapter(f"{output_path}/{args.exp_name}_ccl","ccl")
            
    del model
    del dataloaders
    import time        
    print("Waiting for next language:")
    time.sleep(180)        




if __name__ == "__main__":
    import time

# Start the timer
    start_time = time.time() 
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--all_langs', type=str,help='all langauges for processing')
    parser.add_argument('--result_file', type=str,help='path of the resul file with file name')
    parser.add_argument('--replay_freq', type=int,help='replay frequency')
    parser.add_argument("--replay_type", type = str, help = 'type of replay')
    parser.add_argument("--replay", default =  False, action='store_true', help = "Whether to replay or not")
    parser.add_argument("--UPOS_EVAL", type = str, help = "POS to used for code switch")
    parser.add_argument("--CODE_SWITCH_PATH", type = str, help = "code switch path")
    parser.add_argument("--epochs", type = int, help = "no of epochs")
    parser.add_argument("--codeswitch_type",type = str, help = "code switch type - random/pos" )
    parser.add_argument("--device_id",type = str, help = "gpuid" )
    parser.add_argument("--memory_size",type = int, help = "The size of sample memory" )
    parser.add_argument("--model_path", type = str, help = "model storage path")
    parser.add_argument("--exp_name", type = str, help = "experiment_name")
    parser.add_argument("--task", default =  False, action='store_true', help = "taskadapter")
    parser.add_argument("--lang_cs", type = str, help = "code switch base language")
    parser.add_argument("--acc_boost", default =  False, action='store_true' , help = "Boost the langauge")
    parser.add_argument("--cll_train", default =  False, action='store_true' , help = "Boost the langauge")
    parser.add_argument('--mem_cs_thr', type=int, default=5, help='replay frequency')
    parser.add_argument("--model", type = str, help = "model name")
    parser.add_argument("--code_switch_ratio", type = float, default=0.75, help = "model name")
    parser.add_argument("--final_perf", type = float, default=0, help = "final_performance")
    parser.add_argument("--train_size",type = int, default =10000, help = "The size of sample memory" )
    parser.add_argument("--final_perf_dict", type = str, default="NA", help = "final_performance dictls -l")
    parser.add_argument("--replay_adapter", type = str, default="ccl", help = "replay adapters")
    parser.add_argument("--train_lang_adapter", default =  False, action='store_true' , help = "train language adapters")
    parser.add_argument("--train_from_scratch",type = str, default="th_bn", help = "adapters to train from scratch")
    parser.add_argument("--roberta_adapters", default =  False, action='store_true' , help = "roberta adapters")
    parser.add_argument("--only_lang_adapters", default =  False, action='store_true' , help = "only lang adapters and no task adapter")
    parser.add_argument("--en_cs_percent", type = float, default=1, help = "english codeswitch percentage")
    parser.add_argument("--batch_size", type = int, default=16, help = "BATCH SIZE")
    #parser.add_argument("--inter_cs_memory", type=str, default =  "v1", help = "inter cs memory")
                    


    args = parser.parse_args()
    print(args)
    end_time = time.time()
    try:
        train_model(args)
    except:
        import traceback
        traceback_string = traceback.print_exc()
        traceback.print_exc()
        output_path = args.model_path+"/"+args.all_langs
        
        JSON_FILE = output_path+"/"+args.exp_name+".json"
        f = open(JSON_FILE, 'w')
        args_dict = vars(args)
        print()
        args_dict["error"] = traceback_string
        import datetime
        import time
        ts = time.time()
        time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        args_dict["time_stamp"] = time_stamp
        json.dump(args_dict, f, indent=4)  
        f.close()    
    execution_time = end_time - start_time
    print(f"total time in seconds: {execution_time}, in minutes:{execution_time/60}")
    import torch
    torch.cuda.set_device(int(args.device_id))
    torch.cuda.empty_cache()
    import time        
    print("Waiting for the memory to clean up")
    time.sleep(180)          
