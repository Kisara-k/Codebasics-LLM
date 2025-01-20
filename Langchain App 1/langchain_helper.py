from _keys import openai_api_key
import os
os.environ['OPENAI_API_KEY'] = openai_api_key

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

name_template = PromptTemplate(
    input_variables = ['cuisine'],
    template = 'I want to open a restaurant for {cuisine} food. Suggest a fancy name for this.'
)

name_chain = LLMChain(
    llm = OpenAI(temperature=0.7),
    prompt = name_template,
    output_key = 'restaurant_name'
)


item_template = PromptTemplate(
    input_variables = ['restaurant_name'],
    template = "Suggest some menu items from {restaurant_name}. Only a list of item names."
)

item_chain = LLMChain(
    llm = OpenAI(temperature=0.8),
    prompt = item_template,
    output_key = 'menu_items'
)

chain = SequentialChain(
    chains = [name_chain, item_chain],
    input_variables = ['cuisine'],
    output_variables = ['restaurant_name', 'menu_items'],
)

def generate(cusine):
    return chain({'cuisine': cusine})