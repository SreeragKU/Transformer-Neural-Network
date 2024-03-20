import torch
import numpy as np
from transformer import Transformer
from torch.utils.data import Dataset, DataLoader
from torch import nn

english_file = 'dataset/en-ml/train.en'
malayalam_file = 'dataset/en-ml/train.ml'

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

malayalam_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', 'ˌ',
                        'ഁ', 'ം', 'ഃ', 'അ', 'ആ', 'ഇ', 'ഈ', 'ഉ', 'ഊ', 'ഋ', 'ഌ', 'എ', 'ഏ', 'ഐ', 'ഒ', 'ഓ', 'ഔ',
                        'ക', 'ഖ', 'ഗ', 'ഘ', 'ങ',
                        'ച', 'ഛ', 'ജ', 'ഝ', 'ഞ',
                        'ട', 'ഠ', 'ഡ', 'ഢ', 'ണ',
                        'ത', 'ഥ', 'ദ', 'ധ', 'ന',
                        'പ', 'ഫ', 'ബ', 'ഭ', 'മ',
                        'യ', 'ര', 'റ', 'ല', 'ള', 'ഴ', 'വ', 'ശ', 'ഷ', 'സ', 'ഹ',
                        '഼', 'ഽ', 'ാ', 'ി', 'ീ', 'ു', 'ൂ', 'ൃ', 'ൄ', 'െ', 'േ', 'ൈ', 'ൊ', 'ോ', 'ൌ', '്', 'ൎ', 'ൗ', 'ൠ',
                        'ൡ',
                        '൦', '൧', '൨', '൩', '൪', '൫', '൬', '൭', '൮', '൯', PADDING_TOKEN, END_TOKEN]

english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      ':', '<', '=', '>', '?', '@',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                      'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                      'Y', 'Z',
                      '[', '\\', ']', '^', '_', '`',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                      'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z',
                      '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

index_to_malayalam = {k: v for k, v in enumerate(malayalam_vocabulary)}
malayalam_to_index = {v: k for k, v in enumerate(malayalam_vocabulary)}
index_to_english = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index = {v: k for k, v in enumerate(english_vocabulary)}

with open(english_file, 'r', encoding='utf-8') as file:
    english_sentences = file.readlines()
with open(malayalam_file, 'r', encoding='utf-8') as file:
    malayalam_sentences = file.readlines()

# Limit Number of sentences
TOTAL_SENTENCES = 1000000
english_sentences = english_sentences[:TOTAL_SENTENCES]
malayalam_sentences = malayalam_sentences[:TOTAL_SENTENCES]
english_sentences = [sentence.rstrip('\n').lower() for sentence in english_sentences]
malayalam_sentences = [sentence.rstrip('\n') for sentence in malayalam_sentences]
max_sequence_length = 200


def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True


def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1)  # need to re-add the end token so leaving 1 space


valid_sentence_indicies = []
for index in range(len(malayalam_sentences)):
    malayalam_sentence, english_sentence = malayalam_sentences[index], english_sentences[index]
    if is_valid_length(malayalam_sentence, max_sequence_length) \
            and is_valid_length(english_sentence, max_sequence_length) \
            and is_valid_tokens(malayalam_sentence, malayalam_vocabulary):
        valid_sentence_indicies.append(index)

malayalam_sentences = [malayalam_sentences[i] for i in valid_sentence_indicies]
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]

import torch

d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
ml_vocab_size = len(malayalam_vocabulary)

transformer = Transformer(d_model,
                          ffn_hidden,
                          num_heads,
                          drop_prob,
                          num_layers,
                          max_sequence_length,
                          ml_vocab_size,
                          english_to_index,
                          malayalam_to_index,
                          START_TOKEN,
                          END_TOKEN,
                          PADDING_TOKEN)
class TextDataset(Dataset):

    def __init__(self, english_sentences, malayalam_sentences):
        self.english_sentences = english_sentences
        self.malayalam_sentences = malayalam_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.malayalam_sentences[idx]


dataset = TextDataset(english_sentences, malayalam_sentences)

batch_size = 3
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)

for batch_num, batch in enumerate(iterator):
    if batch_num > 3:
        break


def tokenize(sentence, language_to_index, start_token=True, end_token=True):
    sentence_word_indicies = [language_to_index[token] for token in list(sentence)]
    if start_token:
        sentence_word_indicies.insert(0, language_to_index[START_TOKEN])
    if end_token:
        sentence_word_indicies.append(language_to_index[END_TOKEN])
    for _ in range(len(sentence_word_indicies), max_sequence_length):
        sentence_word_indicies.append(language_to_index[PADDING_TOKEN])
    return torch.tensor(sentence_word_indicies)


eng_tokenized, ml_tokenized = [], []
for sentence_num in range(batch_size):
    eng_sentence, ml_sentence = batch[0][sentence_num], batch[1][sentence_num]
    eng_tokenized.append(tokenize(eng_sentence, english_to_index, start_token=False, end_token=False))
    ml_tokenized.append(tokenize(ml_sentence, malayalam_to_index, start_token=True, end_token=True))
eng_tokenized = torch.stack(eng_tokenized)
ml_tokenized = torch.stack(ml_tokenized)

from torch import nn

criterian = nn.CrossEntropyLoss(ignore_index=malayalam_to_index[PADDING_TOKEN],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NEG_INFTY = -1e9


def create_masks(eng_batch, ml_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    for idx in range(num_sentences):
        eng_sentence_length, ml_sentence_length = len(eng_batch[idx]), len(ml_batch[idx])
        eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
        ml_chars_to_padding_mask = np.arange(ml_sentence_length + 1, max_sequence_length)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, ml_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, ml_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, ml_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        eng_batch, ml_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, ml_batch)
        optim.zero_grad()
        ml_predictions = transformer(eng_batch,
                                     ml_batch,
                                     encoder_self_attention_mask.to(device),
                                     decoder_self_attention_mask.to(device),
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=True)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(ml_batch, start_token=False, end_token=True)
        loss = criterian(
            ml_predictions.view(-1, ml_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == malayalam_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"English: {eng_batch[0]}")
            print(f"Malayalam Translation: {ml_batch[0]}")
            ml_sentence_predicted = torch.argmax(ml_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in ml_sentence_predicted:
              if idx == malayalam_to_index[END_TOKEN]:
                break
              predicted_sentence += index_to_malayalam[idx.item()]
            print(f"Malayalam Prediction: {predicted_sentence}")


            transformer.eval()
            ml_sentence = ("",)
            eng_sentence = ("should we go to the mall?",)
            for word_counter in range(max_sequence_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, ml_sentence)
                predictions = transformer(eng_sentence,
                                          ml_sentence,
                                          encoder_self_attention_mask.to(device),
                                          decoder_self_attention_mask.to(device),
                                          decoder_cross_attention_mask.to(device),
                                          enc_start_token=False,
                                          enc_end_token=False,
                                          dec_start_token=True,
                                          dec_end_token=False)
                next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_malayalam[next_token_index]
                ml_sentence = (ml_sentence[0] + next_token, )
                if next_token == END_TOKEN:
                  break

            print(f"Evaluation translation (should we go to the mall?) : {ml_sentence}")
            print("-------------------------------------------")