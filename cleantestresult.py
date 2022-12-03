"""
Convert JSONL test output to txt file to use with EASSE

"""

# ISSUES
# See note below on "step three"

import os
import re

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

PMPT = "Remove extraneous spaces from the INPUT. INPUT:\n\n<<text>>\n\n OUTPUT:\n\n"

#Step one, create text file
with open('testresult.jsonl') as f:
    lines = f.readlines()
    for line in lines:
        txt = re.sub('"','',line)
        txt = txt.splitlines()[0].replace('\\n',' ').strip()
        with open('GAME_TEST','a') as out_file:
            out_file.write(txt+'\n')

#Step two, use Openai to get rid of extra spaces in the text
with open('GAME_TEST') as f:
    for line in f:
        print(line)
        cmpl = openai.Completion.create(model='text-davinci-002',prompt=PMPT.replace('<<text>>',line),max_tokens=2000,temperature=0)
        write_txt = cmpl.choices[0].text.strip().replace('\n','')+'\n'
        print(write_txt)
        with open('GAME_TEST_PUNCT_FIX','a') as new_f:
            new_f.write(write_txt)

#Step three: 
# We also manually replaced 31 instances of '\' with '"'
# Step one above probably introduced the naked '\' as it removed the double quotes 


