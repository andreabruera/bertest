from data.mybloodyplots import mybloodyplots
from transformers import BertModel, BertTokenizer
import torch
import numpy
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
import os
from torch.nn import CosineSimilarity
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib
import matplotlib._color_data as mcd

bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
cos = CosineSimilarity(dim=-1, eps=1e-6)
os.makedirs('results', exist_ok=True)

datasets = ['men.txt', 'simlex999.txt']
#datasets = ['simlex999.txt']
golden = mcd.CSS4_COLORS['goldenrod']
teal = mcd.CSS4_COLORS['steelblue']
colors = [teal, golden]
current_font_folder = '/import/cogsci/andrea/fonts'

for dataset in datasets:

    word_vectors = defaultdict(list)

    print('Now collecting the golden words and judgements...')

    with open(os.path.join('data', dataset)) as input_file:
        lines = [l for l in input_file.readlines()]
        judgements = []
        dataset_name = re.sub('\.txt', '', dataset)
        dataset_words = []
        if dataset_name == 'simlex999':
            lines = lines[1:]
        for line in lines:
            line = [re.sub('-.', '', k) for k in line.strip().split()]
            if dataset_name == 'simlex999':
                judgements.append([line[0], line[1], line[3]])
            else:
                judgements.append(line)
            words = line[:2]
            for w in words:
                if w not in dataset_words:
                    dataset_words.append(w)
    word2index = {k : i for i, k in enumerate(dataset_words)}
    print('Now collecting the word vectors...')

    for w in tqdm(dataset_words):

        input_id = torch.tensor([bert_tokenizer.encode(w)])
        with torch.no_grad():
            hidden_layers = bert_model(input_id)[2]
        for layer_index, layer in enumerate(hidden_layers):

            if len(input_id[0]) == 3:
                layer_tensor = layer[0][1]
            else:
                slice_length = len(input_id[0])-2
                layer_tensor = torch.zeros(768,)
                for index in range(slice_length):
                    layer_tensor += layer[0][index+1]
            if layer_index == 0:
                layer_index = 'input'
            word_vectors[layer_index].append(layer_tensor)
            del layer_tensor

    golden_scores = [float(k[2]) for k in judgements]

    cosine_plot = []
    r_plot = []
    with open('results/results_{}.txt'.format(dataset_name), 'w') as o:
        for layer_name, layer in word_vectors.items():
            print('Now calculating correlations for layer: {}'.format(layer_name))
            o.write('Results for layer {}\n\n'.format(layer_name))
            
            predicted_cos = []
            predicted_r = []
            for j in judgements:
                w_one = word_vectors[layer_name][word2index[j[0]]]
                w_two = word_vectors[layer_name][word2index[j[1]]]
                prediction = cos(w_one, w_two).item()
                predicted_cos.append(prediction)
                predicted_r.append(spearmanr(w_one.numpy(), w_two.numpy())[0])

            cosine_layer = spearmanr(predicted_cos, golden_scores)[0]
            cosine_plot.append(cosine_layer)
            r_layer = spearmanr(predicted_r, golden_scores)[0]
            r_plot.append(r_layer)
            print('\tCorrelation for the cosine scores: {}'.format(cosine_layer))
            o.write('\tCorrelation for the cosine scores: {}\n'.format(cosine_layer))
            print('\tCorrelation for the correlation scores: {}'.format(r_layer))
            o.write('\tCorrelation for the correlation scores: {}\n\n\n'.format(r_layer))
    
    results_plots = mybloodyplots.MyBloodyPlots(output_folder='results', font_folder=current_font_folder, x_variables=[n for n in word_vectors.keys()], y_variables=[(cosine_plot[i], r_plot[i]) for i in range(len(r_plot))], x_axis='', y_axis='Layers', labels=['Cosine', 'Spearman correlation'], title='BERT word representation test with the {} dataset'.format(dataset_name.capitalize()), identifier=dataset_name, colors=[teal, golden], x_ticks=True, y_ticks=True, y_grid=True)
    results_plots.plot_dat('two_lines')
