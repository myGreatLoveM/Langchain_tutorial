from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import os
from dotenv import load_dotenv
from argparse import ArgumentParser

load_dotenv()

parser = ArgumentParser()
parser.add_argument("--language", default="python")
parser.add_argument("--task", default="return a list of few random numbers")
args = parser.parse_args()

llm = OpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    model_name = 'text-davinci-003', 
    temperature=0.6
)

code_generation_prompt = PromptTemplate(
    input_variables=['language', 'task'],
    template="Write a very short {language} function that will {task}."
)

code_test_prompt = PromptTemplate(
    input_variables=['language', 'code'],
    template="Write a test case for the following {language} code : \n {code}"
)

code_generation_chain = LLMChain(
    llm=llm,
    prompt=code_generation_prompt,
    output_key='code'
)

code_test_chain = LLMChain(
    llm=llm,
    prompt=code_test_prompt,
    output_key='test'
)

chain = SequentialChain(
    chains=[code_generation_chain, code_test_chain],
    input_variables=['language', 'task'],
    output_variables=['code', 'test']
)

output = chain({
    'language': args.language,
    'task': args.task
})

print('>>>>>>> Genearted Code : ')
print(output['code']),

print('>>>>>>> Genearted test : ')
print(output['test'])
