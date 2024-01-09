from ast import literal_eval
import math
#import evaluate

#rouge = evaluate.load('rouge')

import os
import openai
#from keybert import KeyLLM, KeyBERT
#from keybert.llm import OpenAI as KeyBertOpenAI

#from sentence_transformers import SentenceTransformer

openai.api_key = os.environ.get("OPENAI_API_KEY")

# keybert_llm = KeyBertOpenAI()
# sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# kw_model = KeyBERT(model=sentence_model) #KeyLLM(keybert_llm)

def convert_to_lst(l):
    """convert stringified lst to lst"""
    try:
        return literal_eval(l)
    except Exception:
        return l


def flatten_extend_1d(matrix):
    """flatten the list to 1 dimension"""
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list


def tag_based_filter_1dsubpillars(lst, tag_name):
    """Filter"""
    if isinstance(lst, float) and math.isnan(lst):
        return False
    return tag_name in lst


def flatten_extend_2d(matrix):
    """flatten the list to 1 dimension"""
    flat_list = []
    for row in matrix:
        if len(row) == 0:
            continue
        flat_list.extend(row)
    return flat_list


def tag_based_filter_2dsubpillars_matrix(lst, tag_name):
    """ Filter """
    if isinstance(lst, float) and math.isnan(lst):
        return False
    return tag_name in (tuple(item) for item in lst)

def tag_based_filter_sectors_2dsubpillars(lst, tag_name):
    """ Filter """
    if isinstance(lst, float) and math.isnan(lst):
        return False

    return tag_name in flatten_extend_2d(lst)

# def summarization_evaluation(input_excerpts_lst, summaries_lst):
#     """ Calculate rouge score """
#     return rouge.compute(
#         predictions=summaries_lst,
#         references=input_excerpts_lst,
#         rouge_types=["rouge1", "rouge2", "rougeL"],
#         use_aggregator=False
#     )

def text_cleanups(text):
    """ Basic cleanup of the texts """
    text = text.replace("\n", "")
    return text.strip()

# def get_text_embeddings(text):
#     """ Returns the text embeddings using OpenAI """
#     # return openai.Embedding.create(
#     #     input=text,
#     #     model="text-embedding-ada-002"
#     # )
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = model.encode(text, convert_to_tensor=True)
#     return embeddings

# def get_keywords(text):
#     """ Extracts the keywords from the texts """
#     # return kw_model.extract_keywords(
#     #     text,
#     #     embeddings=embeddings,
#     #     threshold=threshold
#     # )
#     return kw_model.extract_keywords(
#         text,
#         keyphrase_ngram_range=(1, 1)
#     )
