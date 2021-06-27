from preprocessing import *
import torch
from transformers import BertTokenizer, BertModel
import spacy
import json
from scipy.spatial.distance import cosine


class BertEmbeddings:

    def __init__(self, model_name):
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.nlp = spacy.load("en_core_web_sm")
        # self.seed_dict = json.load(seed_path)

    def sentence_split(self, review):
        doc = self.nlp(review)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences

    def get_tokens(self, sentences):
        tokens_tensor_list = []
        segments_tensor_list = []
        for i,sent in enumerate(sentences):
            
            marked_text = "[CLS] " + sent + " [SEP]"
            tokenized_text = self.tokenizer.tokenize(marked_text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            sent_len = len(indexed_tokens)
            segments_ids = [i+1] * sent_len
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensor = torch.tensor([segments_ids])
            tokens_tensor_list.append(tokens_tensor)
            segments_tensor_list.append(segments_tensor)
        return tokens_tensor_list, segments_tensor_list

    def get_sent_embeddings(self,tokens_tensor,segments_tensors):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding

    def get_seed_embeddings(self,pos_seed_list,neg_seed_list):
        
        temp_bigram_tokens_tensor_list, temp_bigram_segments_tensors_list = embeddings.get_tokens(["good"])
        temp_bigram_sent_emb = embeddings.get_sent_embeddings(temp_bigram_tokens_tensor_list[0], temp_bigram_segments_tensors_list[0])
        pos_seed_tokens_tensor_list, pos_seed_segments_tensors_list = embeddings.get_tokens(pos_seed_list)
        pos_seed_sent_emb = embeddings.get_sent_embeddings(pos_seed_tokens_tensor_list[0],pos_seed_segments_tensors_list[0]) #fix the summing
        print('similarity score')
        print(1 - cosine(temp_bigram_sent_emb, pos_seed_sent_emb))
        # for i in range(len(pos_seed_tokens_tensor_list)):
        #     pos_seed_sent_emb = embeddings.get_sent_embeddings(pos_seed_tokens_tensor_list[i],pos_seed_segments_tensors_list[i]) #fix the summing
        neg_seed_tokens_tensor_list, neg_seed_segments_tensors_list = embeddings.get_tokens(neg_seed_list)
        neg_seed_sent_emb = embeddings.get_sent_embeddings(neg_seed_tokens_tensor_list[0],neg_seed_segments_tensors_list[0]) #fix the summing
        # for j in range(len(neg_seed_tokens_tensor_list)):
        #     neg_seed_sent_emb = embeddings.get_sent_embeddings(neg_seed_tokens_tensor_list[0],neg_seed_segments_tensors_list[0]) #fix the summing
        return pos_seed_sent_emb, neg_seed_sent_emb

    def compare_similarity(self,pos_seed_emb,neg_seed_emb,sent_emb):
        pos_score = 1 - cosine(sent_emb, pos_seed_emb)
        neg_score = 1 - cosine(sent_emb, neg_seed_emb)
        identify = pos_score/neg_score
        print(identify)
        if identify>1:
            return 'positive'
        else:
            return 'negative'


if __name__ == "__main__":
    embeddings = BertEmbeddings('bert-base-uncased')
    pos_seed_list = ["bad"]
    neg_seed_list = ["bad"]
    pos_seed_sent_emb, neg_seed_sent_emb = embeddings.get_seed_embeddings(pos_seed_list,neg_seed_list)

    sent_list = []
    id_list = []
    reviews = DataLoader('Data/bgg-15m-reviews.csv')
    pandemic_data, pandemic_review = reviews.process_text('Pandemic')
    for j,review in enumerate(pandemic_review):
        sentences = embeddings.sentence_split(review)
        tokens_tensor_list, segments_tensors_list = embeddings.get_tokens(sentences)
        for i in range(len(tokens_tensor_list)):
            sent_emb = embeddings.get_sent_embeddings(tokens_tensor_list[i],segments_tensors_list[i])
            identity = embeddings.compare_similarity(pos_seed_sent_emb,neg_seed_sent_emb,sent_emb)
            sent_list.append(sentences[i])
            id_list.append(identity)
        if j==5:
            break
    print(sent_list)
    print(id_list)