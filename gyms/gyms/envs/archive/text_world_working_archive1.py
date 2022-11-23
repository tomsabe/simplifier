"""
Gym for Text Simplification

Credits:
https://www.gymlibrary.dev/content/environment_creation/
Spacy
sentence-transformers/all-MiniLM-L6-v2
textstat
gramformer
"""

# ISSUES
# Sentence splitter
# Keep looking at measures of simplicity : word length, sentence length

import os
from cgitb import text
import gym
from gym import spaces
from gym.spaces.utils import flatten_space
import pygame
from pygame.locals import *
import numpy as np
import pandas as pd
import spacy
import textstat
import openai
from gramformer import Gramformer
from sentence_transformers import SentenceTransformer
#from wordhoard import Synonyms
import textwrap
import random

# Initialize the NLP tools
NLP = spacy.load("en_core_web_sm")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',use_auth_token=HF_TOKEN)
GF = Gramformer(models=1,use_gpu=False)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define Pygame parameters
pygame.font.init()
FONT = pygame.font.SysFont(None,32)
WHITE = (255,255,255)
YELLOW = (125,125,0)
BLACK = (0,0,0)

#Define the Gym environment
class TextWorldEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,render_mode=None):
        ''' Initialize the Gym environment '''
        self.width=1920/2       # Pygame window width and height
        self.height=1080/2
        #We have 4 state observations: token_len, lemma_len, token_#ancestors, span_len
        self.observation_space = spaces.MultiDiscrete([20,20,5,30,20])
        #We have 6 actions: 
        # (L)eft, (R)ight, (D)rop word, (S)implify word, (L)exical simplification, (Q)uit
        self.action_space = spaces.Discrete(6)
        self._action_to_keypress = {
            0: 'L',
            1: 'R',
            2: 'D',
            3: 'S',
            4: 'B',
            5: 'Q'
        }
        self._keypress_to_action = {
            'L':0,
            'R':1,
            'D':2,
            'S':3,
            'B':4,
            'Q':5
        }
        #Place the game texts into a dataframe that is easily sampled:
        self.originals = pd.read_json('gametexts.json')
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._total_score = None
        self._last_total_score = None

    def _get_obs(self):
        ''' Get a state observation from the environment '''
        try:
            tok = self._token_list[self._word_cursor]
        except:
            print(f"Error getting token {self._word_cursor} from list {self._token_list}.\n")
        n_ancestors = sum(1 for dummy in tok.ancestors)
        tok_per_sent = int(len(self._token_list)/textstat.sentence_count(self._word_text))
        state = [
                min(len(tok.text),20),
                min(len(tok.lemma_),20),
                min(n_ancestors,5),
                min(len(tok.sent),30),
                min(tok_per_sent,20)
                ]
#        print(f"Observation: {state}")
        return np.array(state)

    def _get_info(self):
        ''' Return any auxilliary information that comes with a step.'''
        text = ''
        word_count = 0
        for word in self._word_list:
            if word_count == self._word_cursor:
                text += ' \033[92m '
            text += (' ' + word + ' ')
            if word_count == self._word_cursor:
                text += ' \033[0m '
            word_count += 1            
        info = {"text":textwrap.fill(text,width=60),
                "simple_score":self._simple_score,
                "semantic_score":self._semantic_score,
                "total_score":self._total_score,
                "start_score":self._start_score}
        return info

    def _embedding(self, text):
        ''' Use an LLM to return a vector encoding of text '''
### OpenAI version of embeddings: 
#        resp = openai.Embedding.create(
#                                model="text-similarity-babbage-001",
#                                input=text
#                                )
#        try: 
#            emb = resp['data'][0]['embedding']
#        except:
#            print('Problem with embedding\n')
#            emb = []
#        return emb
### Also tried: all-MiniLM-L6-v2
        return MODEL.encode(text)
### Also tried simcse, could not install python package
###   May be able to try simcse's huggingface implementation

    def _cosine_similarity(self, doc1, doc2):
        ''' Return cosine similarity of two vectors '''
        cos_sim = np.dot(doc1,doc2)/(np.linalg.norm(doc1)*np.linalg.norm(doc2))
        return int(cos_sim*100)/100        

    def _reset_text_from_word_list(self):
        ''' Copy from word list (a list of tokens) to a text string '''
        text = ''
        for i in range(len(self._word_list)):
            text += self._word_list[i]
            if i<(len(self._word_list)-1):
                if self._word_list[i+1] != '.':
                    text += ' '
        self._word_text = text
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
        pos_y+=32
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
        self._current_emb = self._embedding(self._word_text)
        similarity = self._cosine_similarity(self._current_emb,self._start_emb)
        self._semantic_score = similarity*100-100  
        self._total_score = self._simple_score+self._semantic_score
        return

    def _drop_word(self):
        ''' Drop the selected word from a text '''
        #If words remain, drop current one
        if len(self._word_list)==0:
            return
        del self._word_list[self._word_cursor]
        del self._token_list[self._word_cursor]
        #Reset the word text and update the scores
        self._reset_text_from_word_list()
        if len(self._token_list)==0:
            self._word_cursor=0
        else:
            self._word_cursor = np.random.randint(0,high=len(self._token_list))
        self._update_scores()
        return 

    def _get_synonyms(self,text):
        prompt = f"List synonyms for the word '{text}' that a young child would understand, or say 'None' if none exist."
        try:
            compl = openai.Completion.create(
                                        model="text-davinci-002",
                                        prompt=prompt,
                                        max_tokens=20,
                                        temperature=0
                                        )
        except:
            print("Failure with OpenAI API in _get_synonyms")
        try:
            syn_list_text = compl['choices'][0]['text']
            if ('None' in syn_list_text):
                return ''
            else:
                clean_list = syn_list_text.replace('\n','').replace("'",'').split(',')
                cleaner_list = [item.strip() for item in clean_list]
                cleanest_list = list(set(cleaner_list))
                final_list = cleanest_list
#                final_list = [ item for item in cleanest_list if len(item)<len(text) ]
                if not final_list:
                    final_list = ''
                return final_list
        except:
            print('Something went wrong with synonyms.\n')
        return '' #Returning a string if None or error, consistent with Synonyms pckg convention

    def _simplify_word(self):
        ''' Try to simplify the selected word in a text '''
        #If we have moved to a new word, get a new synonym list
        if self._word_cursor != self._simple_cursor:
#            synonym = Synonyms(search_string=self._word_list[self._word_cursor])
#            self._synonym_results = synonym.find_synonyms()
            self._synonym_results = self._get_synonyms(self._word_list[self._word_cursor])
            #if a string is returned, it is a 'No synonym' message 
            if isinstance(self._synonym_results,str):
                return
#            self._synonym_results.sort(key=lambda x:len(str(x)))
            self._simple_cursor = self._word_cursor
        #If there are (remaining) synonyms for this word try shortest one
        if self._synonym_results:
            try:
#                best_synonym=self._synonym_results.pop() #try shortest ones
                best_synonym=random.choice(self._synonym_results)
            except:
                return
            self._word_list[self._word_cursor]=best_synonym
            #Reset the word text and update the scores
            self._reset_text_from_word_list()
            self._reset_word_list_from_text()
            self._word_cursor = np.random.randint(0,high=len(self._token_list))
            self._update_scores()
        return

#Following does not work very well:
#    def _simple_word(self):
#        word=self._word_list[self._word_cursor]
#        text=self._word_text.replace('\n','').strip()
#        instruction = f'If there exists a simpler word or phrase for "{word}" then use it.'
#        compl = openai.Edit.create(
#                                model="text-davinci-edit-001",
#                                input=text,
#                                instruction=instruction,
#                                temperature=0
#                                )
#        print(f"Text: {text}\n Instruction:{instruction}\n")
#        try:
#            revised_text = compl['choices'][0]['text']
#            #Clean up as needed
#            clean_text = revised_text.replace('\n',' ').strip()
#            print(clean_text)
#            #Reset word text and word cursor; update score:
#            self._word_text = clean_text
##            self._word_cursor = 0
#            self._reset_word_list_from_text()
#            self._word_cursor = np.random.randint(0,high=len(self._token_list))
#            self._update_scores()
#        except:
#            print('Something went wrong with word simplification.\n')
#        return

    def _lex_text(self):
        text=self._word_text
        prompt = f"Split complex sentences into simple sentences.\nTEXT:\n{text}'\n"
        compl = openai.Completion.create(
                                        model="text-davinci-002",
                                        prompt=prompt,
                                        max_tokens=200,
                                        temperature=0
                                        )
        try:
            revised_text = compl['choices'][0]['text']
            #Clean up as needed
            clean_text = revised_text.replace('\n',' ').strip()
            #Reset word text and word cursor; update score:
            self._word_text = clean_text
#            self._word_cursor = 0
            self._reset_word_list_from_text()
            self._word_cursor = np.random.randint(0,high=len(self._token_list))
            self._update_scores()
        except:
            print('Something went wrong with lexical simplification.\n')
        return

    def _correct_grammar(self): 
        # REF: https://github.com/PrithivirajDamodaran/Gramformer/
        corrected = GF.correct(self._word_text,max_candidates=1)
        self._word_text = next(iter(corrected))
        self._reset_word_list_from_text()
        self._word_cursor = 0 #make sure not out of range
        ## QUESTION : Do I need to update the scores here? 
        return

    def reset(self, seed=None, options=None):
        # Need to seed self.np_random
#        super().reset(seed=seed) raised error, unexpected kwarg 'seed'
#        super().reset() - maybe not necessary at all?
        super().reset()
        # Pick a sentence to simplify
        self._start_text = self.originals.sample()['original'].iloc[0]
        self._start_doc = NLP(self._start_text)
        self._start_emb = self._embedding(self._start_text)
        #Initialize the word list and token list
        self._word_list = []
        self._token_list = []
        for token in self._start_doc: 
            self._word_list.append(token.text)
            self._token_list.append(token)
        self._word_text = self._start_text
        #Initialize parameters for the actions
        self._simple_cursor = None
        self._synonym_results = []
        # Set agent to first word
        self._word_cursor = np.random.randint(0,high=len(self._token_list)) # was zero
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
#        print(f"Environment step: {action}")
        terminated = False
        last_total_score = self._total_score
        if action == 'L':  #Move left
            self._word_cursor += -1 #try wrap 
            if self._word_cursor < 0:
                self._word_cursor = len(self._word_list)-1
        elif action == 'R': #Move right
            self._word_cursor += 1
            if self._word_cursor > len(self._word_list)-1:
                self._word_cursor = 0
        elif action == 'S': #Simplify word
            self._simplify_word()
        elif action == 'D': #Drop word
            self._drop_word()
        elif action == 'B': #Break up complex sentences
            self._lex_text()  ## TRY TRAINING WITHOUT THIS STEP, SEE IF WE CAN GET ANY PROGRESS
        elif action == 'Q': #End game
            self._correct_grammar() #Enforce grammar correction
            terminated = True
        #Incremental rewards
        reward = self._total_score - last_total_score
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info
    
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
            self.clock.tick(self.metadata["render_fps"])
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