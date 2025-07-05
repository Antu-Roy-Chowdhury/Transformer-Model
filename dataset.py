import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None :
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tonekizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len  = seq_len

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['SOS'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['EOS'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['PAD'])], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        enc_input_token = self.tokenizer_src_encoder(src_text).ids
        dec_input_token = self.tokenizer_tgt_encoder(tgt_text).ids

        enc_num_padding_token = self.seq_len - len(enc_input_token) -2
        dec_num_padding_token = self.seq_len - len(enc_input_token) -1

        if enc_num_padding_token < 0 or dec_num_padding_token < 0:
            raise ValueError('Sentence is too long')
       # Adding SOS And  Eos to Source Text 
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_token, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *enc_num_padding_token, dtype=torch.init64)
            ]
        )
        # adding SOS to deocder input
        decoder_input = torch.cat(
            [

                self.sos_token,
                torch.tensor([dec_input_token], dtype = torch.int64),
                torch.tensor([self.pad_token] *dec_num_padding_token, dtype=torch.init64)
            ])
        # adding EOS to lebel
        label = torch.cat(
            [
                torch.tensor(dec_input_token, dtype =torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_token, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return{
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input!= self.pad.token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask":(decoder_input != self.pad_token).unsqeeze(0). seqeeze(0).int() & causal_mask(decoder_input.size(0)),
            "lebel": label,
            "src_text": src_text,
            "tgt_text": tgt_text

        }
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask ==0



        

