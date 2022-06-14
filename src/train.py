from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv



def train_model(data_path):
    sts_dataset_path = data_path

    # Read the dataset
    model_name = 'paraphrase-mpnet-base-v2'
    train_batch_size = 16
    num_epochs = 2
    model_save_path = './model/' + model_name + '-' + datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S")

    model = SentenceTransformer(model_name)

    train_samples = []
    dev_samples = []
    test_samples = []
    with open(sts_dataset_path, 'r', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            # if row['split'] == 'dev':
            #     dev_samples.append(inp_example)
            # elif row['split'] == 'test':
            #     test_samples.append(inp_example)
            # else:
            train_samples.append(inp_example)
    split = int(len(train_samples)*.80)
    train_dataloader = DataLoader(train_samples[0:split], shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(train_samples[split:], name='sts-dev')

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

    model.fit (train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,


              warmup_steps=warmup_steps,
              output_path=model_save_path)

    # test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    evaluator(model, output_path=model_save_path)

    return "model saved at " + model_save_path