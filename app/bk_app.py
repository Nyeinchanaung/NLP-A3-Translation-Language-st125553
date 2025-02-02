import streamlit as st
import numpy as np
import pickle
import torch
from helpers.classes import *
from helpers.utils import get_text_transform, my_tokenizer
import torchtext

# Loading the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
meta = pickle.load(open('models/meta-additive.pkl', 'rb'))

# Define Transformers
# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

token_transform = meta['token_transform']
vocab_transform = meta['vocab_transform']
text_transform = get_text_transform(token_transform, vocab_transform)

# Load Model
SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'my'

input_dim   = len(vocab_transform[SRC_LANGUAGE])
output_dim  = len(vocab_transform[TRG_LANGUAGE])
hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1

SRC_PAD_IDX = PAD_IDX
TRG_PAD_IDX = PAD_IDX

enc = Encoder(input_dim, 
              hid_dim, 
              enc_layers, 
              enc_heads, 
              enc_pf_dim, 
              enc_dropout, 
              device)

dec = Decoder(output_dim, 
              hid_dim, 
              dec_layers, 
              dec_heads, 
              dec_pf_dim, 
              enc_dropout, 
              device)

model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.apply(initialize_weights)

# Load the model weights
model.load_state_dict(torch.load("models/Seq2SeqTransformer-additive.pt", map_location=device))

# Streamlit UI
st.title("Translation App")

# Input text box
prompt = st.text_input("Enter text to translate:", "")

if st.button("Translate"):
    if prompt:
        max_seq = 100

        src_text = text_transform[SRC_LANGUAGE](prompt).to(device)
        src_text = src_text.reshape(1, -1)
        text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)
        src_mask = model.make_src_mask(src_text)

        model.eval()
        with torch.no_grad():
            enc_output = model.encoder(src_text, src_mask)
        
        outputs = []
        input_tokens = [EOS_IDX]
        for i in range(max_seq):
            with torch.no_grad():
                starting_token = torch.LongTensor(input_tokens).unsqueeze(0).to(device)
                trg_mask = model.make_trg_mask(starting_token)

                output, attention = model.decoder(starting_token, enc_output, trg_mask, src_mask)

            pred_token = output.argmax(2)[:, -1].item()
            input_tokens.append(pred_token)
            outputs.append(pred_token)
            
            if pred_token == EOS_IDX:
                break
        
        trg_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[i] for i in outputs]
        translated_text = " ".join(trg_tokens[1:-1])

        st.success(f"Translated Text: {translated_text}")
    else:
        st.warning("Please enter some text to translate.")