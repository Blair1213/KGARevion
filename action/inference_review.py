import os
import json
import torch
import transformers
from peft import PeftModel
import spacy
import scispacy
from scispacy.linking import EntityLinker
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import difflib
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_utils import SequenceSummary
from sklearn.metrics.pairwise import cosine_similarity
from src.prompter import Prompter
from src.descriptionTemplate import DescriptionTemplate

rel_list = ['protein_protein', 'carrier', 'enzyme', 'target', 'transporter', 'contraindication', 'indication', 'off-label use', 'synergistic interaction', 'associated with', 'parent-child', 'phenotype absent', 'phenotype present', 'side effect', 'interacts with', 'linked to', 'expression present', 'expression absent']

prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given a triple from a knowledge graph. Each triple consists of a head entity, a relation, and a tail entity. Taking (PHYHIP, protein_protein, KIF15) as an example, it means that protein PHYHIP has an interaction with protein KIF15. Please determine the correctness of the triple and response True or False. Please directly output 'True' or 'False'.

### Input:
{}

### Response:

"""

embed_tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')
sequence_summary = SequenceSummary(AutoConfig.from_pretrained('GanjinZero/UMLSBert_ENG'))
bert = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')

set_seed(42)

class ReviewInfer(object):
    def __init__(self, model = None, tokenizer = None, model_weights = None, model_name = None):
        super(ReviewInfer, self).__init__()
        if model is not None and tokenizer is not None:
            self.tokenizer = tokenizer
            self.tokenizer.pad_token_id = tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"  # Allow batched inference
            self.model = self.load_model(model, model_weights)
        elif model is None and model_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
            self.tokenizer.pad_token_id = tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"  # Allow batched inference
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map='auto')
            self.model = self.load_model(model, model_weights)

        self.umls_to_ddb = self.read_primekg_umls()
        embedding_path = model_weights + "embeddings.pth"
        lm_to_kg_path = model_weights + "lm_to_kg.pth"
        self.kg_embeddings = torch.load(embedding_path, map_location='cuda:0')
        self.lm_to_kg = torch.load(lm_to_kg_path, map_location='cuda:0')
        self.prompter = Prompter("alpaca")
        self.descriptionTemp = DescriptionTemplate()
        
        self.rel_dict = self.rel_dict_primeKG()
        self.nlp, self.linker = self.load_entity_linker()
        
    def read_primekg_umls(self):
        umls_to_ddb = {}
        with open('primeKG/primeKG_to_umls_cui.csv') as f:
            elms = pd.read_csv(f)
            entity_id_primekg = elms['entity_index']
            cui_code = elms['cui']
            for idx, e in enumerate(entity_id_primekg):
                umls_to_ddb[cui_code[idx]] = e
        return umls_to_ddb
    
    def load_model(self, model, model_weights):
        model = PeftModel.from_pretrained(model, model_weights).cuda()
        model.config.pad_token_id = self.tokenizer.eos_token_id 
        model = model.eval()

        return model
    
    def load_entity_linker(self, threshold=0.90):
        nlp = spacy.load("en_core_sci_sm")
        linker = EntityLinker(
            resolve_abbreviations=True,
            name="umls",
            threshold=threshold)
            
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        
        return nlp, linker
    
    def entity_linking_to_umls(self, entity):
        doc = self.nlp(entity)
        entities = doc.ents
        all_entities_results = []
        
        for mm in range(len(entities)):
            entity_text = entities[mm].text
            entity_start = entities[mm].start
            entity_end = entities[mm].end
            all_linked_entities = entities[mm]._.kb_ents
            all_entity_results = []
            for ii in range(len(all_linked_entities)):
                curr_concept_id = all_linked_entities[ii][0]
                curr_score = all_linked_entities[ii][1]
                curr_scispacy_entity = self.linker.kb.cui_to_entity[all_linked_entities[ii][0]]
                curr_canonical_name = curr_scispacy_entity.canonical_name
                curr_TUIs = curr_scispacy_entity.types
                curr_entity_result = {"Canonical Name": curr_canonical_name, "Concept ID": curr_concept_id,
                                  "TUIs": curr_TUIs, "Score": curr_score}
                all_entity_results.append(curr_entity_result)
            curr_entities_result = {"text": entity_text, "start": entity_start, "end": entity_end,
                                "start_char": entities[mm].start_char, "end_char": entities[mm].end_char,
                                "linking_results": all_entity_results}
            all_entities_results.append(curr_entities_result)
        return all_entities_results
    
    def entitylinker(self, entity):
        def map_to_ddb(ent_obj):
            res = []
            if len(ent_obj) == 0:
                return res
            for ent_cand in ent_obj['linking_results']:
                CUI  = ent_cand['Concept ID']
                name = ent_cand['Canonical Name']
                if CUI in self.umls_to_ddb:
                    ddb_cid = self.umls_to_ddb[CUI]
                    res.append((ddb_cid, name))
            return res
        
        query_ents = self.entity_linking_to_umls(entity)
        if len(query_ents) == 0:
            return None

        def string_similar(s1, s2):
            return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

        qc = []
        qc_name = []
        qc_sim = []
        for ent_obj in query_ents:
            res = map_to_ddb(ent_obj)
            #print("res is {}".format(res))
            if len(res) == 0:
                return None
            for elm in res:
                ddb_cid, name = elm
                qc.append(ddb_cid)
                qc_name.append(name)
                qc_sim.append(string_similar(entity, name))
    
        max_index = qc_sim.index(max(qc_sim))

        return qc[max_index]
    
    def rel_dict_primeKG(self):
        rel_dict = []
        for r in rel_list:
            tokenized = embed_tokenizer.encode_plus(r,
                                          max_length=16,
                                          truncation=True,
                                          padding=True,
                                          add_special_tokens=True)
            input_ids = torch.LongTensor([tokenized['input_ids']])
            attention_mask = torch.LongTensor([tokenized['attention_mask']])

            token_embedding = bert(input_ids, attention_mask)
            rel_embedding = sequence_summary(token_embedding[0])
            rel_dict.append(rel_embedding)
        return rel_dict
    
    def rellinker(self, rel):
        def embedding_similarity(rel_dict, rel_embedding):
            cos_sim = []
            for r in rel_dict:
                cos_sim.append(cosine_similarity(r.detach().numpy(), rel_embedding.detach().numpy())[0][0])
            #print(cos_sim)
            return cos_sim.index(max(cos_sim))

        tokenized = embed_tokenizer.encode_plus(rel,
                                          max_length=16,
                                          truncation=True,
                                          padding=True,
                                          add_special_tokens=True)
        input_ids = torch.LongTensor([tokenized['input_ids']])
        attention_mask = torch.LongTensor([tokenized['attention_mask']])

        token_embedding = bert(input_ids, attention_mask)
        rel_embedding = sequence_summary(token_embedding[0])
        
        return embedding_similarity(self.rel_dict, rel_embedding)


    def score(self, data):

        head_entity, rel, tail_entity = data[:3]
        ent = "\nThe input triple: \n({}, {}, {})\n".format(head_entity.strip(), rel.strip(), tail_entity.strip())

        full_prompt = "Given a triple from a knowledge graph. The triple consists of a head entity, a relation, and a tail entity. Please determine the correctness of the triple and response True or False. Do not provide any explanation or additional context. Only return the letter of the selected option. The output must be one of 'True' or 'False'"
        full_prompt += ent

        head_id = self.entitylinker(head_entity)
        rel_id = self.rellinker(rel)
        tail_id = self.entitylinker(tail_entity)
        
        triplet_description = self.descriptionTemp.get_description([head_entity, rel_list[rel_id], tail_entity])
        desc_token = self.tokenizer(
            triplet_description,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt',
        )

        lm_emb = self.model.model.model.embed_tokens(desc_token['input_ids'].to('cuda'))
        attention_mask = desc_token['attention_mask'].to('cuda')
        
        if head_id == None or tail_id == None:
            return 'True', 0.5
            #kg_lm_emb = lm_emb
        else:
            ids = [head_id, rel_id, tail_id]
            embed_ids = torch.LongTensor(ids).reshape(1, -1).to('cuda')
            kg_emb = self.kg_embeddings(embed_ids)
            kg_lm_emb = self.lm_to_kg(kg_emb, lm_emb, attention_mask)
        #print(kg_lm_emb.shape)
        
        prompt = "<|start_header_id|>system<|end_header_id|>You are a helpful assistant. <|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>{}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|><|eot_id|>"
        prompt = prompt.format(full_prompt)
        tokenized_full_prompt = self.tokenizer(prompt, return_tensors='pt')

        input_ids = tokenized_full_prompt['input_ids'].cuda()

        token_embeds = self.model.model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((kg_lm_emb, token_embeds), dim=1)
        batch_size, seq_len, _ = kg_lm_emb.shape[:3]
        prefix_mask = torch.ones((batch_size, seq_len))
        new_attention_mask = torch.cat((prefix_mask.cuda(), tokenized_full_prompt['attention_mask'].cuda()), dim=-1)
        
        token_logit = self.model(inputs_embeds = input_embeds, attention_mask = new_attention_mask).logits
        true_logit = token_logit[:, -1, 2575]
        false_logit = token_logit[:, -1, 4139]
       
        true_prob = (true_logit / (true_logit + false_logit)).item()
        false_prob = (false_logit / (true_logit + false_logit)).item()

        if true_prob > false_prob:
            return 'True', true_prob
        else:
            return 'False', true_prob
       


if __name__ == '__main__':
    llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map='auto')
    model = ReviewInfer(llm_model, llm_tokenizer, "primekg_r64_alpha_16_bz_256_epoch_1_llama3_lr_0.0003_review_ratio_0.2/")
    print(model.score(['ADH1B', 'protein_protein', 'KIF15']))
    print(model.score(['Clathrin', 'interacts with', 'FAT3 protein']))
    print(model.score(['AHR', 'target', 'TG']))