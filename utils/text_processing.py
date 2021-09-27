import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torchaudio.compliance import kaldi # for mel bins
import unicodedata

class Normalization():
    """This class is for normalizing the spectrograms batch by batch. The normalization used is min-max, two modes 'framewise' and 'imagewise' can be selected. In this paper, we found that 'imagewise' normalization works better than 'framewise'"""
    def __init__(self, mode='framewise'):
        if mode == 'framewise':
            def normalize(x):
                size = x.shape
                x_max = x.max(1, keepdim=True)[0] # Finding max values for each frame
                x_min = x.min(1, keepdim=True)[0]  
                output = (x-x_min)/(x_max-x_min) # If there is a column with all zero, nan will occur
                output[torch.isnan(output)]=0 # Making nan to 0
                return output
        elif mode == 'imagewise':
            def normalize(x):
                size = x.shape
                x_max = x.view(size[0]*size[1]).max(0, keepdim=True)[0]
                x_min = x.view(size[0]*size[1]).min(0, keepdim=True)[0]
                x_max = x_max # Make it broadcastable
                x_min = x_min # Make it broadcastable 
                return (x-x_min)/(x_max-x_min)
        else:
            print(f'please choose the correct mode')
        self.normalize = normalize

    def __call__(self, x):
        return self.normalize(x)
    
    
spec_normalize = Normalization()    






class Speech_Command_label_Transform:
    def __init__(self, data):
        self.labels = sorted(list(set(datapoint[2] for datapoint in data))) # ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    def label_to_index(self, word):
        word = word.split("/")[-1]
        return self.labels.index(word)
    def index_to_label(self, index):
        return self.labels[index]



def speech_command_processing(data, Speech_Command_label_transform, input_key='waveform', label_key='utterance', downsample_factor=320):
    waveforms = []
    labels = []

    for batch in data:
        waveforms.append(batch[0].squeeze(0)) 
        label = Speech_Command_label_transform.label_to_index(batch[2])
        labels.append(label)
                
    waveform_padded = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    labels = torch.Tensor(labels)

    output_batch = {'waveforms': waveform_padded, 
             'labels': labels
             }
    return output_batch



class TextTransform:
    """Maps characters to integers and vice versa
       mode: char or word.
       When char is used, break down the texts by characters.
       When word is used, break down the texts by words (break at space)
       When ph is used, break down the texts by phonemics which is same as word"""
    def __init__(self, ipa_dict, mode):
        self.ipa_dict = ipa_dict
        self.mode = mode
        reverse_ipa_dict = {}
        for key, value in ipa_dict.items():
            reverse_ipa_dict[value] = key
        self.reverse_ipa_dict = reverse_ipa_dict
        if mode == 'char':
            self.text_to_int = self._char_to_int
            self.int_to_text = self._int_to_char            
        elif mode == 'word' or mode == 'ph':
            self.text_to_int = self._word_to_int
            self.int_to_text = self._int_to_word

    def _word_to_int(self, ipa_sequence):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = [] 

        # split() should be used instead of spilt(' ')
        # This prevents ' ' froming appearing as the character
        # Hence prevents 0 from appearing at the end of the token 
        for c in ipa_sequence.split(): 
    #         for c in ipa_sequence.replace('  ', ' ').split(' '): 
            ch = self.ipa_dict[c]
            int_sequence.append(ch)
        return int_sequence, ipa_sequence
    
    def _char_to_int(self, ipa_sequence):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = [] 

        # split() should be used instead of spilt(' ')
        # This prevents ' ' froming appearing as the character
        # Hence prevents 0 from appearing at the end of the token 
        for c in ipa_sequence: 
    #         for c in ipa_sequence.replace('  ', ' ').split(' '): 
            ch = self.ipa_dict[c]
            int_sequence.append(ch)
        return int_sequence, ipa_sequence
    
    def _int_to_word(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.reverse_ipa_dict[i]+' ') # seperating each phoneme/word with a space
        return ''.join(string).replace('<SPACE>', '').replace('  ',' ') # Remove <SPACE>
    
    def _int_to_char(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.reverse_ipa_dict[i]) # appending each character back to the sentence
        return ''.join(string)

       

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.reverse_ipa_dict[i]+' ') # seperating each phoneme with a space
        return ''.join(string).replace('<SPACE>', '').replace('  ',' ') # Remove <SPACE>
#         return ''.join(string).replace('  ',' ')


def data_processing(data, text_transform, input_key='waveform', label_key='utterance', downsample_factor=320):
    waveforms = []
    labels = []
    input_lengths = []
    label_lengths = []
    utterance = []
    path = []
    for batch in data:
        waveforms.append(batch[input_key].squeeze(0)) # remove batch dim
#         ipa_sequence = phonemize(utterance.lower(),
#                              language=language,
#                              backend='espeak',
#                              strip=True,
#                              language_switch='remove-flags',
#                              separator=separator.Separator(phone=" ", word=" <SPACE> "))
#         try:
        tokens, ipa_sequence = text_transform.text_to_int(batch[label_key])
#         except Exception as e:
#             print(' '*100)
#             print(e)
#             print(f"batch['path'] = {batch['path']}")
        utterance.append(batch[label_key])
        label = torch.Tensor(tokens)
        
        labels.append(label)
        # modify this according to the model downsampling rate
        input_lengths.append(batch[input_key].shape[1]//downsample_factor-1)
        
        label_lengths.append(len(label))
        
        path.append(batch['path'])
        
        waveform_padded = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)

        
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    
    output_batch = {'waveforms': waveform_padded, # it is waveforms instead of spectrograms, this tiny hack can make the code work with existing training function
             'labels': labels,
             'input_lengths': torch.tensor(input_lengths), 
             'label_lengths': torch.tensor(label_lengths),
             'ipa_utterance': utterance,
             'path': path
             }
    return output_batch


def GreedyDecoder(output, labels, label_lengths, text_transform, blank=0):
    """faster better version"""
    predictions = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    PERs = []
    for i, pred in enumerate(predictions):
        decode = []
        pred = pred.unique_consecutive()# remove repeated predictions
        pred = pred[pred!=blank] # remove blanks (default: 0)
        
#         dist, _, counter = edit_distance(labels[i].tolist()[:label_lengths[i]], pred.tolist())
#         PERs.append((counter['sub'] + counter['ins'] + counter['del'])/counter['words'])
        
        decodes.append(text_transform.int_to_text(pred.cpu().numpy()))
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        
    return decodes, targets#, PERs


def edit_distance(ref, hyp):
    assert isinstance(ref, list) and isinstance(hyp, list)

    dist = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.uint32)
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                dist[0][j] = j
            elif j == 0:
                dist[i][0] = i
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dist[i][j] = dist[i - 1][j - 1]
            else:
                substitute = dist[i - 1][j - 1] + 1
                insert = dist[i][j - 1] + 1
                delete = dist[i - 1][j] + 1
                dist[i][j] = min(substitute, insert, delete)

    i = len(ref)
    j = len(hyp)
    steps = []
    while True:
        if i == 0 and j == 0:
            break
        elif i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1] and \
            ref[i - 1] == hyp[j - 1]:
            steps.append('corr')
            i, j = i - 1, j - 1
        elif i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1] + 1:
            assert ref[i - 1] != hyp[j - 1]
            steps.append('sub')
            i, j = i - 1, j - 1
        elif j >= 1 and dist[i][j] == dist[i][j - 1] + 1:
            steps.append('ins')
            j = j - 1
        else:
            assert i >= 1 and dist[i][j] == dist[i - 1][j] + 1
            steps.append('del')
            i = i - 1
    steps = steps[::-1]

    counter = Counter({'words': len(ref), 'corr': 0, 'sub': 0, 'ins': 0,
        'del': 0})
    counter.update(steps)

    return dist, steps, counter


def GreedyDecoder_slow(output, labels, label_lengths, text_transform, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets