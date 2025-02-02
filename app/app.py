import streamlit as st
import torch
import pickle
from helpers.classes import Encoder, Decoder, Seq2SeqTransformer, initialize_weights
from helpers.utils import get_text_transform, my_tokenizer

# Load the Meta Model (Tokenization & Vocabulary)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
meta = pickle.load(open("models/meta-additive.pkl", "rb"))

# Extract token and vocabulary transforms
token_transform = meta["token_transform"]
vocab_transform = meta["vocab_transform"]
text_transform = get_text_transform(token_transform, vocab_transform)

# Define language identifiers
SRC_LANGUAGE = "en"
TRG_LANGUAGE = "my"

# Define Special Tokens
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

# Get vocab mappings
mapping = vocab_transform[TRG_LANGUAGE].get_itos()

# Define Model Parameters
input_dim = len(vocab_transform[SRC_LANGUAGE])
output_dim = len(vocab_transform[TRG_LANGUAGE])
hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1

# Build Model
enc = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device)
dec = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, device)
model = Seq2SeqTransformer(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
model.apply(initialize_weights)

# Load Pretrained Weights
model.load_state_dict(torch.load("models/Seq2SeqTransformer-additive.pt", map_location=device))
model.eval()


def beam_search(model, src_text, src_mask, beam_size=5, max_len=50):
    sequences = [[[], 0.0]]  # Store (token list, score)

    with torch.no_grad():
        enc_output = model.encoder(src_text, src_mask)

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            input_tokens = [SOS_IDX] + seq  # Ensure it starts correctly
            starting_token = torch.LongTensor(input_tokens).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(starting_token)

            with torch.no_grad():
                output, _ = model.decoder(starting_token, enc_output, trg_mask, src_mask)

            top_k = output[:, -1, :].topk(beam_size)

            for i in range(beam_size):
                token = top_k.indices[0][i].item()
                prob = top_k.values[0][i].item()
                new_seq = seq + [token]
                new_score = score + prob

                # Stop adding tokens if EOS is encountered
                if token == EOS_IDX:
                    all_candidates.append([new_seq, new_score])
                    break  
                
                all_candidates.append([new_seq, new_score])

        sequences = sorted(all_candidates, key=lambda x: (len(x[0]), x[1]), reverse=True)[:beam_size]

    # Return the longest valid sequence before EOS
    final_tokens = sequences[0][0]
    if EOS_IDX in final_tokens:
        final_tokens = final_tokens[:final_tokens.index(EOS_IDX)]  # Cut at EOS

    return final_tokens



# Streamlit UI
st.title("English-to-Myanmar Translator")
#st.write("Enter an English sentence below to get its Myanmar translation.")

# Input box
prompt = st.text_input("Enter text in English:")

if st.button("Translate"):
    if prompt:
        src_text = text_transform[SRC_LANGUAGE](prompt).to(device).reshape(1, -1)
        src_mask = model.make_src_mask(src_text)

        # Perform beam search translation
        output_tokens = beam_search(model, src_text, src_mask, beam_size=5)

        # Convert token IDs to words and remove <eos>
        translated_text = " ".join(mapping[token] for token in output_tokens if token < len(mapping) and token != EOS_IDX)

        if translated_text.strip():
            st.success(f"**Translation:** {translated_text}")
        else:
            st.warning("Translation resulted in an empty output. Try modifying your input.")
    else:
        st.warning("Please enter text to translate.")

