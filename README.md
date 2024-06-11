# Chatbot_Langchain_Llama2

This medical chatbot leverages advanced AI technologies to provide users with accurate and relevant medical information. Utilizing LangChain to manage interactions and Pinecone as a vector database for efficient similarity search, the chatbot processes user queries and retrieves relevant data. The powerful LLaMA 2 language model generates responses based on the context of the queries. The front-end is designed using HTML and CSS to create a modern, user-friendly interface similar to popular messaging applications. Users can ask health-related questions, seek guidance on medical procedures, and get information about medications, all through an intuitive chat interface. This combination of cutting-edge technology and thoughtful design ensures a seamless and responsive user experience, making reliable medical information easily accessible.

![image](https://github.com/AkhilaKamma/Chatbot_Langchain_Llama2/assets/22701124/4a599345-e183-47ee-a4a6-4f1459cbdd73)


This Medical chatbot generates reliable and up to date information


### STEP 01- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

# Medical chatbot
![image](https://github.com/AkhilaKamma/Chatbot_Langchain_Llama2/assets/22701124/51340477-1c06-4f02-80a1-5c8dfb7196f9)


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone

