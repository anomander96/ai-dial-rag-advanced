from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role

SYSTEM_PROMPT = """
    You are a RAG-powered assistant that assists users with their questions about microwave usage.
                
    ## Structure of User message:
    `RAG CONTEXT` - Retrieved documents relevant to the query.
    `USER QUESTION` - The user's actual question.

    ## Instructions:
    - Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
    - Cite specific sources when using information from the context.
    - Answer ONLY based on conversation history and RAG context.
    - If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""


USER_PROMPT = """
    # RAG CONTEXT
    {context}

    # USER QUESTION
    {query}
"""

DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
}

embeddings_client = DialEmbeddingsClient(
    deployment_name = 'text-embedding-3-small-1',
    api_key = API_KEY
)

chat_client = DialChatCompletionClient(
        deployment_name = "gpt-4o",
        api_key = API_KEY
    )

text_processor = TextProcessor(
    embeddings_client = embeddings_client,
    db_config = DB_CONFIG
)


def run_chat():
    print("🎯 Microwave RAG Assistant (type 'exit' to quit)")
    
    conversation = Conversation()
    conversation.add_message(Message(role = Role.SYSTEM, content = SYSTEM_PROMPT))

    while True:
        # get user input
        user_input = input("\n").strip()

        if user_input.lower() == 'exit':
            print("End of conversation")
            break

        # Retrieval step: find relevant chunks from DB
        results = text_processor.search(
            search_mode = SearchMode.COSINE_DISTANCE,
            query = user_input,
            top_k = 4,
            min_score = 0.5
        )

        # build a context string from search results
        context = "\n\n".join([row['text'] for row in results])

        # Augmentation step
        augmented_prompt = USER_PROMPT.format(context = context, query = user_input)

        # add user message to the conversation history
        conversation.add_message(Message(role = Role.USER, content = augmented_prompt))

        # Generation step
        response = chat_client.get_completion(conversation.messages)
        conversation.add_message(response)

        print(f"\nAI: {response.content}")



run_chat()