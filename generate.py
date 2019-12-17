###############################################################################
# Language Modeling on Literature Books
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import data
import os
import json
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model_storage/model-t2l4h.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='./samples/generated_temp1.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1337,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--beam', type=int, default='1',
                    help='number of beams in the search algorithmn')
args = parser.parse_args()

def beam_search_decoder(candidates, candidates_score, b, output):
    """
    candidates: Torch Tensor
    canditates_score: Numpy array
    b: int
    output: Torch Tensor
    """
    high_indices=np.zeros((b,b),dtype=np.int64)
    probs_sampled=np.zeros((b,b))
    indices = torch.multinomial(output,b)
    for i in range(b):
        probs_sampled[i,:]= output[i,:][indices[i]].cpu().numpy()
        if candidates_score is not None:
            probs_sampled[i,:] = np.log(probs_sampled[i,:]) + candidates_score[i]
        else:
            probs_sampled[i,:] = np.log(probs_sampled[i,:])       
    
    top_indices = probs_sampled.flatten().argsort()[-b:][::-1]
    scores=probs_sampled.flatten()[top_indices]
    high_indices=indices.view(-1)[top_indices.tolist()]
    
    new_candidates=torch.zeros(candidates.size(),dtype=torch.int64).to(device)
    for i in range(b):
        aux=int(top_indices[i]/b)
        new_candidates[:,i]=candidates[:,aux]
    high_indices=high_indices.view(1,-1).to(device)
    return torch.cat([new_candidates,high_indices],dim=0) , scores

# Set the random seed manually for reproducibility.
#torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

with open(os.path.join(args.data, 'data3.json'), 'r') as fp:
    data_dict = json.load(fp)
corpus = data.LM_Dataset(data_dict)
ntokens = len(corpus.encoder._all_subtoken_strings)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

if args.beam > 1:
    if not is_transformer_model:
        hidden = model.init_hidden(args.beam)
    candidates = torch.randint(ntokens, (1, args.beam), dtype=torch.long).to(device)
    candidates_score = None

subtoken_ids=[]
with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            if args.beam ==1:
                if is_transformer_model:
                    output = model(input, False)
                    word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                    input = torch.cat([input, word_tensor], 0)
                else:
                    output, hidden = model(input, hidden)
                    word_weights = output.squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    input.fill_(word_idx)

                #word = corpus.dictionary.idx2word[word_idx]
                subtoken_ids.append(word_idx.item())

                #outf.write(word + ('\n' if i % 20 == 19 else ' '))

                if i % args.log_interval == 0:
                    print('| Generated {}/{} subtokens'.format(i, args.words))
            else:
                if is_transformer_model:
                    output = model(candidates, False)
                    output = output[-1].squeeze().div(args.temperature).exp().cpu()
                    candidates , candidates_score = beam_search_decoder(candidates, candidates_score, args.beam, output)

                else:
                    output, hidden = model(candidates[-1,:].view(1,-1), hidden, apply_softmax= True)
                    output = output.squeeze()
                    candidates , candidates_score = beam_search_decoder(candidates, candidates_score, args.beam, output)
            
        if args.beam > 1:
            subtoken_ids=candidates[:,0].cpu().numpy().tolist()
                
        outf.write(corpus.encoder.decode(subtoken_ids))
