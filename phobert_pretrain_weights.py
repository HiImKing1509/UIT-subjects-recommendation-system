from transformers import AutoModel, AutoTokenizer

def load_phobert_pretrain():
    phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    return phobert, tokenizer