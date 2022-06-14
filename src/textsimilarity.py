from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, LoggingHandler, losses, models, util
import nltk
import faiss
import numpy as np
import pandas as pd
import sentence_transformers
import  os

print(sentence_transformers.__version__)
print(nltk.__version__)
print(faiss.__version__)
print(pd.__version__)
print(np.__version__)



word_embedding_model = sentence_transformers.models.Transformer('./model/paraphrase-mpnet-base-v2-2021-07-07_13-50-47/0_Transformer')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])




def get_similarity_dir(dir,doc):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dir):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    if doc.split(".")[-1].lower() == 'txt':
        with open(doc,'r') as f:
            doc1 = f.readlines()

        doc1 = nltk.sent_tokenize(' '.join(doc1))

        orig_file = []
        matched_file = []
        matched_sen = []
        score = []
        orig_sen = []
        for file in listOfFiles:

            with open(file, 'r') as f:
                doc2 = f.readlines()

            doc2 = nltk.sent_tokenize(' '.join(doc2))
            doc2_embeddings = model.encode(doc2, show_progress_bar=True, convert_to_numpy=True)
            embedding_size = 768  # Size of embeddings

            # Defining our FAISS index
            # Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
            n_clusters = len(doc2)
            # We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
            quantizer = faiss.IndexFlatIP(embedding_size)
            index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
            # Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.
            index.nprobe = 3
            ### Create the FAISS index
            print("Start creating FAISS index")
            # First, we need to normalize vectors to unit length
            doc2_embeddings = doc2_embeddings / np.linalg.norm(doc2_embeddings, axis=1)[:, None]
            # Then we train the index to find a suitable clustering
            index.train(doc2_embeddings)
            # Finally we add all embeddings to the index
            index.add(doc2_embeddings)

            doc1, doc2, sc = find_similarity(doc1, doc2, index)

            if len (doc1) > 0 :
                for i,d1 in enumerate(doc1):

                    orig_file.append(doc)
                    matched_file.append(file)
                    orig_sen.append(d1)
                    matched_sen.append(doc2[i])
                    score.append(sc[i])

    df = pd.DataFrame({"orig_file" : orig_file,"matched_file":matched_file,"orig_sen":orig_sen,"matched_sen":matched_sen,"score":score})
    return df

def find_similarity(doc1,doc2,index):



    top_k_hits = 10
    orig = []
    dup = []
    sc = []

    for d1 in doc1:

        inp_question = d1

        # inp_question =  ' '.join(inp_question.split())
        # print(inp_question)
        question_embedding = model.encode(inp_question)

        # FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal to cosine similarity
        question_embedding = question_embedding / np.linalg.norm(question_embedding)
        question_embedding = np.expand_dims(question_embedding, axis=0)

        # Search in FAISS. It returns a matrix with distances and corpus ids.
        distances, corpus_ids = index.search(question_embedding, top_k_hits)

        # We extract corpus ids and scores for the first query
        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)

        # print("Input question:", inp_question)

        for hit in hits[0:top_k_hits]:
            if hit['score'] > 0.55:
                orig.append(d1)
                dup.append(doc2[hit['corpus_id']])
                sc.append(hit['score'])

                print("{}\t{}\t{:.3f}\t".format(d1, doc2[hit['corpus_id']], hit['score']))

        return orig,dup,sc


def get_similarity(file1,file2):

    with open(file1,'r') as f:
        doc1 = f.readlines()

    with open(file2,'r') as f:
        doc2 = f.readlines()

    doc1 = nltk.sent_tokenize(' '.join(doc1))
    doc2 = nltk.sent_tokenize(' '.join(doc2))



    print("Encode the corpus. This might take a while")
    # doc1_embeddings = model.encode(doc1, show_progress_bar=True, convert_to_numpy=True)
    doc2_embeddings = model.encode(doc2, show_progress_bar=True, convert_to_numpy=True)

    embedding_size = 768    #Size of embeddings
             #Output k hits

    #Defining our FAISS index
    #Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    n_clusters = len(doc2)
    #We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
    #Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.
    index.nprobe = 3
    ### Create the FAISS index
    print("Start creating FAISS index")
    # First, we need to normalize vectors to unit length
    doc2_embeddings = doc2_embeddings / np.linalg.norm(doc2_embeddings, axis=1)[:, None]
    # Then we train the index to find a suitable clustering
    index.train(doc2_embeddings)
    # Finally we add all embeddings to the index
    index.add(doc2_embeddings)

    doc1,doc2,sc =find_similarity(doc1,doc2,index)

    df = pd.DataFrame({"doc1" : doc1,"doc2":doc2,"score":sc})
    return df
