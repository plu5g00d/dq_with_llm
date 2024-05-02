import streamlit as st
from openai import OpenAI
import pandas as pd

## session_state
## openai_model
## idx
## prompt_messages

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

st.title("LLM for DQ Rules - using openai")

idx = -1


if "prompt_messages" not in st.session_state:
    system_prompt = '''You are a data scientist. 
        Your task is to thoroughly analyze the provided dataset and recommend rules to check data quality.
        Please limit your response to 10 lines.
        '''
    st.session_state.prompt_messages = [{"role": "system", "content": system_prompt}]

#### load and display dataset

def load_data(dataset_file):
    data = pd.read_csv(dataset_file)
    lowercase = lambda x: str(x).lower().strip()
    data.rename(lowercase, axis= 'columns', inplace = True)
    return data

dataset_pizza = load_data('pizza_sales.csv')
dataset_coursera = load_data('coursera_course_dataset_v2_no_null.csv')
SHOW_N_ROWS = 15
sample_dataset_pizza = dataset_pizza[:SHOW_N_ROWS]
sample_dataset_coursera = dataset_coursera[:SHOW_N_ROWS]
if st.sidebar.button("Pizza Sales"):
    st.write(sample_dataset_pizza)
if st.sidebar.button("Coursera Catalog"):
    st.write(sample_dataset_coursera)

#### construct prompts
def return_std_prompts(sample_dataset, context):
    qs = []

    #prompt = '''How many questions has ChatGPT answered ? Can you answer questions about yourself ?'''
    #qs.append(prompt)

    ## prompt - 1 - generic
    prompt = '''You are provided with data delimited by triple backticks.
                The data is in a csv format. 
                The column names are provided in the first line. 
                The data has information about {}.'''.format(context)
    prompt = prompt + """"'''
    {}
    '''""".format(sample_dataset)
    prompt = prompt + "Recommend data quality checks for all columns of this dataset."
    qs.append(prompt)

    ## prompt - 2 - data type
    prompt = '''Analyse the columns for data type by looking at the data in each column and provide checks 
    relevant to the data type of the column. 
    '''
    qs.append(prompt)

    ## prompt - 3 - example rule
    prompt = '''
    Rule 1 - If table A has a text column B with fixed set of values 'abc','def','ghi', one check for column B is that 
    the value should belong to the subset 'abc','def' or 'ghi'. Any other text value example 'aei' in column B is a 
    data quality issue. 
    Rule 2 - If table C has a numeric or text column D and no 2 values in column D are the same, one check for column D is 
    that values in column D should be unique.
    In the Pizza sales dataset, for each column look at the data and check for distinct values. Check if Rule 1 and/or 
    2 are applicable. List the column names to with Rule 1 or Rule 2 can be applied. 
    If Rule 1 is applicable, list the enumerated values with the column name.
    '''
    qs.append(prompt)

    ## prompt - 4 - profiling data
    prompt = '''You are given the following information about the {} dataset.
        There are 50,000 rows in the dataset. 
        Column pizza_size has 3 distinct values 'S', 'M', 'L'.
        Column pizza_id has distinct values for all rows.
        Column pizza_category has distinct values 'Classic', 'Veggie', 'Supreme', 'Chicken'.
        Columns quantity,unit_price,total_price have numeric values.
        Provide a data quality check for Column pizza_size, pizza_category and pizza_id.
        '''.format(context)

    qs.append(prompt)

    ## prompt - 5 - drools
    prompt = '''
        Use the knowledge you gained about data quality rules and 
        provide data quality rules for Pizza sales dataset in JSON format 
        as per drools. Display the output as syntax highlighted code markdown.
        '''.format(context)

    qs.append(prompt)

    ## prompt - 6 - learning
    prompt = '''
                The dataset that follows has information about training courses on Coursera and skills learned. 
                Use the knowledge you gained about data quality rules and create data quality rules 
                for columns with enumerated values and for columns with unique values 
                in JSON format as per Drools rule engine. Display the output in JSON markup format.'''

    prompt = prompt + """"'''
    {}
    '''""".format(dataset_coursera)
    qs.append(prompt)

    ## prompt - 7 - goodbye
    prompt = "Say thank you for watching the demo & invite the viewer to learn more about LLMs."
    qs.append(prompt)

    return qs

#### send prompts to LLM
def perform_preset_analysis(idx, qs, prompt_messages):

    prompt = qs[idx]
    with st.chat_message("user"):
        st.markdown(prompt)
        user_q = {"role": "user", "content": prompt}
        prompt_messages.append(user_q)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(model="gpt-3.5-turbo",
                messages =  prompt_messages,
                stream=True,
                temperature=0.2)

    response = st.write_stream(stream)
    a = { "role" : "assistant" , "content" : response }
    prompt_messages.append(a)

#### get prompts for pizza dataset
qs = return_std_prompts(sample_dataset_pizza, 'Pizza Sales')
#st.markdown(qs)

#### send prompts to LLM
on = st.toggle('Activate DQ')
if on:
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    else:
        st.session_state.idx += 1
    st.button('Send prompt ?', key='prompt_send_button',
                on_click=perform_preset_analysis(st.session_state.idx, qs, st.session_state.prompt_messages),
                disabled=False)

