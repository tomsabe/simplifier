"""
Gym for Text Simplification

Credits:
https://www.gymlibrary.dev/content/environment_creation/
OpenAI GPT-3
Spacy
sentence-transformers/all-MiniLM-L6-v2 -  may not use
textstat
"""

# ISSUES
# Implement lookup thesaurus
# Fine tune a sentence splitter
# Study Asset Corpus for optimum simplicity measure
# Keep looking at measures of simplicity : word length, sentence length

import os
import sys
import copy
import warnings
import gym
from gym import spaces
import pygame
from pygame.locals import *
import numpy as np
import pandas as pd
import spacy
import textstat
import openai
import torch
#from gramformer import Gramformer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging
import textwrap

logging.set_verbosity_error() #suppress warnings

# Initialize the NLP tools
NLP = spacy.load("en_core_web_sm")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',use_auth_token=HF_TOKEN)
#GF = Gramformer(models=1,use_gpu=False)
GRAMMAR = AutoModelForSequenceClassification.from_pretrained("tomsabe/autotrain-grammar-check-2154669454", use_auth_token=True)
GRAMMAR_TOK = AutoTokenizer.from_pretrained("tomsabe/autotrain-grammar-check-2154669454", use_auth_token=True)
SOFTMAX = torch.nn.Softmax(dim=1)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Define Pygame parameters
pygame.font.init()
FONT = pygame.font.SysFont(None,32)
WHITE = (255,255,255)
YELLOW = (125,125,0)
BLACK = (0,0,0)

#Define the Gym environment
class TextWorldEnv(gym.Env):

    def __init__(self,render_mode=None):
        ''' Initialize the Gym environment '''
        #Set PyGame width and height
        self.width=1920/2
        self.height=1080/2
        #We have 5 state observations: 
        # token_len, lemma_len, token_#ancestors, span_len, simplicity, # times lex, # times split
        self.observation_space = spaces.MultiDiscrete([20,20,5,30,30,2,2])
        #We have 4 player actions: 
        # (K)eep, (D)rop, (L)exical simplification, (S)yntactic simplification
        self.action_space = spaces.Discrete(4)
        self._action_to_keypress = {
            0: 'K',
            1: 'D',
            2: 'L',
            3: 'S'
        }
        self._keypress_to_action = {
            'K':0,
            'D':1,
            'L':2,
            'S':3
        }
        #Read the game texts into a dataframe that is easily sampled:
        self.originals = pd.read_json('gametexts.json',orient='index').transpose() #adjust the formating
        self._item_number = 0 #only used if texts are to be read sequentially
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._total_score = None
        self._last_total_score = None

    def _get_obs(self):
        ''' Provide a state observation'''
        try:
            tok = self._token_list[self._word_cursor]
        except:
            print(f"Error getting token {self._word_cursor} from list {self._token_list}\n")
        n_ancestors = sum(1 for dummy in tok.ancestors)
        state = [
                min(len(tok.text),20),
                min(len(tok.lemma_),20),
                min(n_ancestors,5),
                min(len(tok.sent),30),
                min(int(self._simple_score/2),30), #compressed version of simplicity score
                self._simp_word_times,
                self._split_times
                ]
        return np.array(state)

    def _get_terminal_obs(self):
        '''Provide dummy state observation if this is terminal step.'''
        state = [0,0,0,0,0,0,0]
        return np.array(state)

    def _get_info(self):
        ''' Return auxilliary information that comes with the state.'''
        #Assemble human-readable text
        text = ''
        word_count = 0
        for word in self._word_list:
            if word_count == self._word_cursor:
                text += ' \033[92m '
            text += (' ' + word + ' ')
            if word_count == self._word_cursor:
                text += ' \033[0m '
            word_count += 1
        #Package aux info into dictionary  
        info = {"text":textwrap.fill(text,width=60),
                "simple_score":self._simple_score,
                "semantic_score":self._semantic_score,
                "total_score":self._total_score,
                "start_score":self._start_score}
        return info

    def _embedding_openai(self, text):
        '''Optional helper - if using OpenAI to provide embeddings'''
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

    def _embedding(self, text, openai=False):
        ''' Use LLM to return a vector encoding of text '''
        if openai:
            return self._embedding_openai(text)
        else:
            return MODEL.encode(text)
        #Could add a simcse option as well 

    def _cosine_similarity(self, doc1, doc2):
        ''' Helper - return cosine similarity of two vectors '''
        cos_sim = np.dot(doc1,doc2)/(np.linalg.norm(doc1)*np.linalg.norm(doc2))
        return int(cos_sim*100)/100       

    def _text_from_word_list(self,word_list):
        ''' Helper - copy from word list to a text string '''
        text = ''
        for i in range(len(word_list)):
            text += word_list[i]
            if i<(len(word_list)-1):
                if word_list[i+1] != '.':
                    text += ' '
        return text

    def _reset_text_from_word_list(self):
        ''' Copy from self word list to self text string '''
        self._word_text = self._text_from_word_list(self._word_list)
        return

    def _reset_word_list_from_text(self):
        ''' Copy from a text string to a list of tokens '''
        doc = NLP(self._word_text)
        self._word_list=[]
        self._token_list=[]
        for token in doc:
            self._word_list.append(token.text)
            self._token_list.append(token)
        return

    def _display_all_words(self, screen):
        ''' Refresh the game text in the GUI '''
        pos_x=0
        pos_y=64
        word_count = 0
        for word in self._word_list:
            if word_count == self._word_cursor:
                text_img = FONT.render(word,True,WHITE,YELLOW)
            else:
                text_img = FONT.render(word,True,WHITE,BLACK)
            screen.blit(text_img,(pos_x,pos_y))
            word_count += 1
            pos_x += (text_img.get_width()+16)
            if pos_x>(self.width-150):
                pos_x=0
                pos_y+=32
        pos_x=0
        pos_y+=48
        text_img = FONT.render(f"Original Text (start score = {self._start_score}):",True,WHITE,BLACK)
        screen.blit(text_img,(pos_x,pos_y))
        pos_y+=32
        for token in self._start_doc:
            text_img = FONT.render(token.text,True,WHITE,BLACK)
            screen.blit(text_img,(pos_x,pos_y))
            pos_x += (text_img.get_width()+16)
            if pos_x>(self.width-150):
                pos_x=0
                pos_y+=32
        return

    def _display_score(self, screen):
        ''' Display the score in the GUI '''
        simple_img = FONT.render(f"Simplicity: {self._simple_score}",True,WHITE)
        simple_rect = simple_img.get_rect()
        screen.blit(simple_img,simple_rect)
        semantic_img = FONT.render(f"  Semantics: {self._semantic_score}",True,WHITE)
        semantic_rect = semantic_img.get_rect()
        semantic_rect.topleft = simple_rect.topright
        screen.blit(semantic_img,semantic_rect)
        total_img = FONT.render(f"  Total: {self._total_score}",True,WHITE)
        total_rect = total_img.get_rect()
        total_rect.topleft = semantic_rect.topright
        screen.blit(total_img,total_rect)
        return

    def _update_scores(self):
        ''' Calculate the game score '''
        self._simple_score=((70-textstat.mcalpine_eflaw(self._word_text)) / 2) \
                            + (30-textstat.flesch_kincaid_grade(self._word_text)) 
        self._simple_score=int(self._simple_score*100)/100
        self._current_emb = self._embedding(self._word_text,openai=True)
        similarity = self._cosine_similarity(self._current_emb,self._start_emb)
        self._semantic_score = similarity*100-100  
        self._total_score = int((self._simple_score+self._semantic_score)*100)/100
        return

    def _drop_word(self):
        ''' Drop the selected word from a text if result will be grammatical '''
        #Only proceed if words remain in list
        if len(self._word_list)==0:
            return
        #Build text without selected word
        drop_list = self._word_list.copy()
        drop_list.pop(self._word_cursor)
        drop_text = self._text_from_word_list(drop_list)
        #Check grammaticality
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = GRAMMAR_TOK(drop_text, return_tensors="pt")
            outputs = GRAMMAR(**inputs)
            #Get probability that grammar is correct
            gram = SOFTMAX(outputs.logits)[0][1]
        #If <90% likely grammar is correct, do not drop but advance cursor
        if gram.item() < 0.9:
            self._word_cursor += 1
            return
        #But if grammar seems okay, go ahead with drop
        del self._word_list[self._word_cursor]
        del self._token_list[self._word_cursor]
        #Reset the word text
        self._reset_text_from_word_list()
        #Update scores
        self._update_scores()
        return 

    def _simplify_words(self):
        ''' Use fine-tune LLM to simplify vocabulary '''
        #Perform the action max twice per game
        if self._simp_word_times > 1:
            self._word_cursor = len(self._word_list)
            return
        text=self._word_text
        prompt = text+'\n\n###\n\n'
        compl = openai.Completion.create(
                                        model="davinci:ft-personal-2022-11-20-15-12-01",
                                        prompt=prompt,
                                        max_tokens=200,
                                        temperature=0,
                                        stop='###'
                                        )
        try:
            revised_text = compl['choices'][0]['text']
            #Clean up as needed
            clean_text = revised_text.replace('\n',' ').strip()
            #Reset word text and cursor; update score:
            self._word_text = clean_text
            self._reset_word_list_from_text() # do this if no grammar check
            self._word_cursor = 0 
            self._update_scores()
            self._simp_word_times += 1
        except:
            print('Something went wrong with simplify words.\n')
        return

    def _split_and_replace_text(self):
        '''Use fine-tune LLM to simplify sentence structure.'''
        #Perform the action max twice per game
        if self._split_times > 1:
            self._word_cursor = len(self._word_list)
            return
        text=self._word_text
        prompt = text+'\n\n###\n\n'
        compl = openai.Completion.create(
                                        model="davinci:ft-personal-2022-11-18-18-36-39",
                                        prompt=prompt,
                                        max_tokens=200,
                                        temperature=0,
                                        stop='###'
                                        )
        try:
            revised_text = compl['choices'][0]['text']
            #Clean up as needed
            clean_text = revised_text.replace('\n',' ').strip()
            #Reset word text and cursor; update score:
            self._word_text = clean_text
            self._reset_word_list_from_text() # do this if no grammar check
            self._word_cursor = 0
            self._update_scores()
            self._split_times += 1
        except:
            print('Something went wrong with split and replace.\n')
        return

    def reset(self, sequential=False):
        super().reset()
        # Pick a sentence to simplify
        if sequential:
            try:
                self._start_text = copy.deepcopy(self.originals.iloc[self._item_number].original) #needs to be a copy
                self._item_number += 1
            except:
                print("End of game texts.\n")
                sys.exit(0)
        else: #pick a random selection
            self._start_text = self.originals.sample()['original'].iloc[0]
        self._start_doc = NLP(self._start_text)
        self._start_emb = self._embedding(self._start_text,openai=True)
        #Initialize the word list and token list
        self._word_list = []
        self._token_list = []
        for token in self._start_doc: 
            self._word_list.append(token.text)
            self._token_list.append(token)
        self._word_text = self._start_text
        #Initialize parameters for the actions
        self._simp_word_times = 0   #keep track how many times words simplified
        self._split_times = 0       #keep track how many times sentences split
        # Set agent to first word
        self._word_cursor = 0
        #Initialize the scores
        self._update_scores()
        self._start_score = self._total_score
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action_code):
        action = self._action_to_keypress[action_code]
        terminate = False
        last_total_score = self._total_score
        if action == 'K':  #'KEEP' - do nothing
            self._word_cursor += 1
            cost = 0
        elif action == 'D': #'DROP' - remove word
            self._drop_word()
            cost = 0
        elif action == 'L': #'Lexical Simplification' 
            self._simplify_words()
            cost = 0
        elif action == 'S': #'Syntactic Simplification' 
            self._split_and_replace_text()
            cost = 0
        #Calculate action reward and get latest info
        reward = self._total_score - last_total_score + cost
        info = self._get_info()
        #if we just passed the last word, return reward and terminate game
        if self._word_cursor == len(self._word_list):
            terminate = True
            observation = self._get_terminal_obs()
            return observation, reward, terminate, False, info
        #Otherwise also report the new observation and render again for human
        observation = self._get_obs()
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminate, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        #Draw to the canvas
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill(BLACK)
        self._display_all_words(canvas)
        self._display_score(canvas)
        if self.render_mode == "human":
            self.window.blit(canvas,canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(4)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)),axes=(1,0,2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    world = TextWorldEnv(render_mode="human")
    world.reset()
    running = True
    while running:
        for event in pygame.event.get():
            #Check for action
            if event.type == QUIT:
                world.close()
            if event.type == KEYDOWN:
                try:
                    action = world._keypress_to_action[event.unicode]
                    _1, _2, term, _4, _5 = world.step(action)
                    if term: 
                        world.reset()
                except: 
                    pass 

#Replay code idea from https://colab.research.google.com/github/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb#scrollTo=owOVWB158NlF
'''
def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(eval_py_env.render())
  return embed_mp4(filename)

create_policy_eval_video(agent.policy, "trained-agent")
'''

# If using Gramformer ... 
#    def _correct_grammar(self): 
#        # REF: https://github.com/PrithivirajDamodaran/Gramformer/
#        corrected = GF.correct(self._word_text,max_candidates=1)
#        self._word_text = next(iter(corrected))
#        self._reset_word_list_from_text()
#        self._word_cursor = 0 #make sure not out of range
#        return

