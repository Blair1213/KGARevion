import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple

from transformers import LlamaForCausalLM
from process_kge import load_pretrain_kge


class ReviewBasic(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM
    ) -> None:
        super(ReviewBasic, self).__init__()
        self.llama_model = model
        # self.embeddings = nn.Embedding(100, 4096)
        self.embeddings = PrefixKGEmbedding(
            num_ent=2034,
            num_rel=42,
            dim_llm=4096,
            num_prefix=1
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        embedding_ids: torch.LongTensor = None
    ):
        kg_embeds = self.embeddings(embedding_ids)
        batch_size, seq_len, _ = kg_embeds.shape
        token_embeds = self.llama_model.model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)
        prefix_mask = torch.ones((batch_size, seq_len))
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)
        new_attention_mask = torch.cat((prefix_mask.cuda(), attention_mask), dim=-1)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)
        return self.llama_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        

class Review(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        num_prefix: int,
        hidden_dim: int,
        kge_model: str,
        pretrain_emb_path = None
    ) -> None:
        super(Review, self).__init__()
        self.llama_model = model
        self.max_length = 256
        ##for 8B 4096, while for 70B it shold be 

        self.device = self.llama_model.device
        self.lm_to_kg = LmToKG(num_layers = 1, hidden_dim = hidden_dim, embed_dim = 8192, device=self.device)
        
        ent_embs, rel_embs = load_pretrain_kge(kge_model)
        #ent_embs.to(self.device)
        #rel_embs.to(self.device)
        if pretrain_emb_path is None:
            print("Adapter Trained From Scratch".format(pretrain_emb_path))
            self.embeddings = PretrainKGEmbedding(
                pretrain_ent_embs=ent_embs,
                pretrain_rel_embs=rel_embs,
                dim_llm=8192,
                num_prefix=num_prefix
            )
            self.embeddings.to(self.device)
        else:
            print("Adapter Load From {}".format(pretrain_emb_path))
            self.embeddings = torch.load(pretrain_emb_path)
            self.embeddings.to(self.device)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        embedding_ids: torch.LongTensor = None
    ):  
        triplet_description_tokens = input_ids[:,:self.max_length]
        triplet_description_attnetion_mask = attention_mask[:,:self.max_length]
        description_embeds = self.llama_model.model.model.embed_tokens(triplet_description_tokens) ##(batch_size, 512, 4096)
        kg_embeds = self.embeddings(embedding_ids) ##kg_embeds.shape (batch_size, 3, 4096)
       
        kg_embeds_lm = self.lm_to_kg(kg_embeds, description_embeds, triplet_description_attnetion_mask)


        batch_size, seq_len, _ = kg_embeds_lm.shape
        token_embeds = self.llama_model.model.model.embed_tokens(input_ids[:, self.max_length:])
        input_embeds = torch.cat((kg_embeds_lm, token_embeds), dim=1) ##相当于增加了3个tokens

        prefix_mask = torch.ones((batch_size, seq_len))
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)
        new_attention_mask = torch.cat((prefix_mask.cuda(), attention_mask[:, self.max_length:]), dim=-1)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)

        return self.llama_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class LmToKG(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        embed_dim,
        device
    ):
        super(LmToKG, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        #self.lin1 = nn.Linear(embed_dim * 2, hidden_dim)
        #self.lin2 = nn.Linear(hidden_dim, embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, embed_dim)
        )

        self.dropout = torch.nn.Dropout(0.5)
        self.layernorm = torch.nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.to(device=device)
    
    def to(self, device):
        for f in self.ffn:
            f.to(device)
        self.dropout.to(device)
        self.layernorm.to(device)
    
    def forward(self, kg_emb, lm_emb, attention_mask = None):
        ##kg_emb shape: [bz, 3, 4096]
        ##lm_emb shape: [bz, 512, 4096]
        ##attention_mask: [bz, 512]
        #print("kg_emb")
        #print(kg_emb.shape)
        #print("lm_emb")
        #print(lm_emb.shape)
        
        logits = torch.matmul(kg_emb, torch.permute(lm_emb, (0, 2, 1)))  ##(bz, 3, 512)
        ##print("logits")
        #print(logits.shape)
        #print("logits on {}".format(logits.device))

        if attention_mask is not None:
            attention_mask = attention_mask.float()
            while attention_mask.dim() < logits.dim():
                attention_mask = attention_mask.unsqueeze(1)
        
        logits = logits + (attention_mask + 1e-45).log()
        logits_lm_to_kg = torch.nn.functional.log_softmax(logits, dim=1) ##(bz, 3, 512)
        #logits_kg_to_lm = torch.nn.functional.log_softmax(logits, dim=-1) ##(bz, 3, 512)

        lm_kg_emd = torch.matmul(logits_lm_to_kg, lm_emb) ##(bz, 3, 4096)
        #kg_lm_emb = torch.matmul(torch.permute(logits_kg_to_lm, (0, 2, 1)), kg_emb) ##(bz, 512, 4096)

        #lm_kg_emd_new = torch.cat([kg_emb, lm_kg_emd], dim=-1) ##(bz, 3, 4096*2)

        out1 = kg_emb + lm_kg_emd

        #print("out1 is on {}".format(out1.device))

        residual = out1
        out1_norm = self.layernorm(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout(ffn_output)
        out2 = residual + ffn_output

        return out2 ##(bz, 3, 4096)



class PrefixKGEmbedding(nn.Module):
    def __init__(
        self,
        num_ent,
        num_rel,
        dim_llm,
        num_prefix
    ):
        super(PrefixKGEmbedding, self).__init__()
        self.emb_dim = num_prefix * dim_llm
        self.ent_embeddings = nn.Embedding(num_ent, self.emb_dim)
        self.rel_embeddings = nn.Embedding(num_rel, self.emb_dim)
    

    def forward(self, triple_ids):
        head, relation, tail = triple_ids[:, 0], triple_ids[:, 1], triple_ids[:, 2]
        h = self.ent_embeddings(head)
        r = self.rel_embeddings(relation)
        t = self.ent_embeddings(tail)
        prefix = torch.stack((h, r, t), dim=1)
        return prefix

class PretrainKGEmbedding(nn.Module):
    def __init__(
        self,
        pretrain_ent_embs,
        pretrain_rel_embs,
        dim_llm,
        num_prefix
    ):
        super(PretrainKGEmbedding, self).__init__()
        self.num_prefix = num_prefix
        self.llm_dim = dim_llm
        self.emb_dim = num_prefix * dim_llm
        self.ent_embeddings = nn.Embedding.from_pretrained(pretrain_ent_embs)
        self.rel_embeddings = nn.Embedding.from_pretrained(pretrain_rel_embs)
        self.pretrain_dim = self.ent_embeddings.weight.shape[1]
        # Froze the pretrain embeddings
        self.ent_embeddings.requires_grad_(False)
        self.rel_embeddings.requires_grad_(False)
        self.adapter = nn.Linear(self.pretrain_dim, self.emb_dim)
    

    def forward(self, triple_ids):
        # main training stage
        if triple_ids.shape[1] == 3:
            head, relation, tail = triple_ids[:, 0], triple_ids[:, 1], triple_ids[:, 2]
            h = self.ent_embeddings(head)
            r = self.rel_embeddings(relation)
            t = self.ent_embeddings(tail)
            pretrain_embs = torch.stack((h, r, t), dim=1)
            prefix = self.adapter(pretrain_embs).reshape(-1, 3*self.num_prefix, self.llm_dim)  ##这里面的num_prefix,相当于每个实体/关系对应几个token
            return prefix
        # entity-aware pre-funing
        else:
            ent = triple_ids.reshape(-1,)
            emb = self.ent_embeddings(ent)
            prefix = self.adapter(emb).reshape(-1, self.num_prefix, self.llm_dim)
            # print(prefix.shape)
            return prefix