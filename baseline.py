""" Less is More : Baseline analysis RL text simplification """

#ISSUES: 
# Need to revise analysis using OpenAI embeddings

import spacy
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import textstat
from dataloader import load_asset_valid
import matplotlib.pyplot as plt
import openai

# Define the NLP models to be used - needs to match text_world.py
NLP = spacy.load("en_core_web_sm")
#HF_TOKEN = os.getenv("HF_TOKEN")
#MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',use_auth_token=HF_TOKEN)
openai.api_key = os.getenv("OPENAI_API_KEY")

#Helper functions for similarity, needs to match implementation in text_world.py
def _embedding(text):
    ''' Use an LLM to return a vector encoding of text '''
    ### For now, using OpenAI version of embeddings: 
    resp = openai.Embedding.create(
                                model="text-similarity-babbage-001",
                                input=text
                                )
    try: 
        emb = resp['data'][0]['embedding']
    except:
        print('Problem with embedding\n')
        emb = []
    return emb
### Also tried: all-MiniLM-L6-v2
#        return MODEL.encode(text)
### Also tried simcse, could not install python package
###   May be able to try simcse's huggingface implementation

def _cosine_similarity(doc1, doc2):
    cos_sim = np.dot(doc1,doc2)/(np.linalg.norm(doc1)*np.linalg.norm(doc2))
    return int(cos_sim*100)/100        

# Helper functions for simplicity
def fkgl_map(text):
    return textstat.flesch_kincaid_grade(text)

def standard_map(text):
    return textstat.text_standard(text)

def mcalpine_map(text):
    return textstat.mcalpine_eflaw(text)

def summary_stat_validation():
    "Give simplicity measures on original and human simplifications"
    vdf = load_asset_valid()
    #Examine Flesch-Kincaid Grade Level
    fkgl_vdf = vdf.applymap(fkgl_map)
    print("Textstat Flesch-Kincaid Grade Level Summary:\n")
    print(fkgl_vdf.describe())
    #Examine Readability Consensus
    standard_vdf = vdf.applymap(standard_map)
    print("Textstat Readability Summary:\n")
    print(standard_vdf.describe())
    #Examine McAlpine EFLAW Readability Score
    mcalpine_vdf = vdf.applymap(mcalpine_map)
    print("Textstat McAlpine Summary:\n")
    print(mcalpine_vdf.describe())
    fig = plt.figure()
    plt1 = fig.add_subplot(311)
    plt2 = fig.add_subplot(313)
    #Plot Flesch-Kincaid improvements
    y_fkgl = []
    for col in fkgl_vdf.columns:
        if col != 'valid.orig':
            plt1.scatter(fkgl_vdf['valid.orig'], fkgl_vdf[col], label=col)
            plt1.legend(loc='best', fontsize=6)
            plt1.set_xlabel('Original FKGL')
            plt1.set_ylabel('Simplified FKGL')
    #Plot McAlpine improvements
    for col in standard_vdf.columns:
        if col != 'valid.orig':
            plt2.scatter(mcalpine_vdf['valid.orig'], mcalpine_vdf[col], label=col)
            plt2.legend(loc='best', fontsize=6)
            plt2.set_xlabel('Original McAlpine')
            plt2.set_ylabel('Simplified McAlpine')
    plt.show()
    return

def semantic_preservation():
    "Give semantic preservation measures on original and human simplifications"
    #Take a sample of 30 
    vdf = load_asset_valid().sample(n=30)
    print("Calculating embeddings ... ")
    emb_sim = vdf.applymap(_embedding)
    orig = emb_sim['valid.orig']
    for i in range(11):
        comp = emb_sim[f'valid.simp.{i}']
        similarity = []
        for j in range(len(vdf)):
            similarity.append(_cosine_similarity(orig.iloc[j],comp.iloc[j]))
        emb_sim[f'sim{i}'] = similarity
    print("Sample Similarity Summary:\n")
    print(emb_sim.describe())

def calculate_scores():
    "Calculate combined score improvement on simlicity and semantic preservation"
    #Take a sample of 30 
    vdf = load_asset_valid().sample(n=30)
    #Calculate FKGL
    print("Calculating FKGL ... ")
    fkgl_vdf = vdf.applymap(fkgl_map)
    #Calculate McAlpine EFLAW Readability Score
    print("Calculating McAlpine EFLAW")
    mcalpine_vdf = vdf.applymap(mcalpine_map)
    #Calculate similarity
    print("Calculating embeddings ... ")
    emb_sim = vdf.applymap(_embedding)
    print(emb_sim)
    orig = emb_sim['valid.orig'].copy()
    emb_sim['valid.orig']=[1 for item in range(len(vdf))]
    for i in range(11):
        comp = emb_sim[f'valid.simp.{i}']
        new_col = []
        for j in range(len(vdf)):
            new_col.append(_cosine_similarity(orig.iloc[j],comp.iloc[j]))
        emb_sim[f'valid.simp.{i}'] = new_col
    print(emb_sim)
    #calculate final score
    simple_df = fkgl_vdf.multiply(-1).add(30).add(mcalpine_vdf.multiply(-1).add(70).multiply(0.5))
    total_df = simple_df.add(emb_sim.multiply(100).add(-100))
    #calculate the improvement
    orig = total_df['valid.orig'].copy()
    total_df['valid.orig']=[0 for item in range(len(vdf))]
    for i in range(11):
        comp = total_df[f'valid.simp.{i}']
        new_col = comp - orig
        total_df[f'valid.simp.{i}'] = new_col
    print(total_df)
    print(total_df.describe())

if __name__ == "__main__":
#    summary_stat_validation()
#    semantic_preservation()
    calculate_scores()
