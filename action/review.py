import logging
from src.utils import BaseLLM
from .generate import TripletExtraction
from .revise import Revise
from .inference_review import ReviewInfer


class Review(object):
    def __init__(self, llm, args) -> None:
        super().__init__()
        self.class_name = 'Review'
        self.llm = llm
        self.args = args
        self.action_desc = 'Using this action to score generated triplets.'
        if self.llm.llm_name in ['llama3.1', 'llama3']:
            self.model = ReviewInfer(model = self.llm.llm_model, tokenizer=self.llm.llm_tokenizer, model_weights = args.weights_path)
        else:
            self.model = ReviewInfer(model_weights = args.weights_path, model_name = 'llama3.1')
        self.is_revise = args.is_revise
        if self.is_revise == True: 
            self.revise = Revise(self.llm)
        self.max_round = args.max_round
        self.triple_generator = TripletExtraction(llm=self.llm, args=args, model_name=args.llm_name, device='cuda')
    
    def check_triplets(self, keys_text):
        match = keys_text.replace('[', '').replace(']', '').replace('\'', '')
        triplets_list = match.split(',')
        split_length = 3
        split_lists = [triplets_list[i:i+split_length] for i in range(0, int((len(triplets_list)/split_length))*split_length , split_length)]

        return split_lists
    
    def output_format(self, output):
        return str(output)
    
    def call(self, triplets, query):
        triplet_list = self.check_triplets(triplets)
        scores = []
        select_triplets = []
        
        for t in triplet_list:
            t = [i.strip() for i in t]
            classification, prob = self.model.score(t)
            logging.info("{} is {}".format(t, classification))
            if classification == 'True':
                select_triplets.append(t)
                scores.append(prob)
            elif classification == 'False' and self.is_revise:
                temp = [t]
                round_num = 0
                while round_num < self.max_round:
                    logging.info("current round is {} and triplets are {}".format(round_num, temp))
                    
                    modified_triple = self.revise.call(temp, query)
                    modified_triple_list = self.check_triplets(modified_triple)
                    
                    for m in modified_triple_list:
                        m = [e.strip() for e in m]
                        m_class, m_prob = self.model.score(m)
                        logging.info("revised triplet {} is {}".format(m, m_class, m_prob))
                        if m_class == 'True':
                            select_triplets.append(m)
                            scores.append(m_prob)
                            round_num = self.max_round
                            break
                        temp.append(m)
                    round_num += 1

        return self.output_format(select_triplets), scores