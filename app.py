from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load local Ollama model
llm = ChatOllama(model="llama3.2:3b")

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. Give simple and clear answers."
    ),
    (
        "human",
        "{question}"
    )
])

# Create output parser
parser = StrOutputParser()

# Build LangChain chain
chain = prompt | llm | parser


# Test the chain with 5 inputs
questions = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is an embedding?",
    "What is LangChain?",
    "What is Ollama?"
]

print("=" * 60)
print("       W6D2 - LANGCHAIN CHAIN TEST")
print("=" * 60)

for i, question in enumerate(questions, 1):
    print(f"\nTest {i}: {question}")

    response = chain.invoke({
        "question": question
    })

    print("Answer:")
    print(response)

print("\n" + "=" * 60)
print("Task 1 completed: Chain tested with 5 inputs")
print("=" * 60)
# ============================================================
# TASK 2 - CONVERSATION MEMORY
# ============================================================

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create chat history
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Create a separate prompt for conversation
memory_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Use the conversation history to answer questions accurately."
    ),
    (
        "placeholder",
        "{history}"
    ),
    (
        "human",
        "{question}"
    )
])

# Create memory chain
memory_chain = memory_prompt | llm | parser

# Add conversation history
conversation_chain = RunnableWithMessageHistory(
    memory_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

print("\n" + "=" * 60)
print("       W6D2 - CONVERSATION MEMORY TEST")
print("=" * 60)

conversation = [
    "My name is Srusti.",
    "What is my name?",
    "I am learning artificial intelligence.",
    "What am I learning?",
    "Can you tell me my name and what I am learning?"
]

for i, question in enumerate(conversation, 1):

    print(f"\nTurn {i}: {question}")

    response = conversation_chain.invoke(
        {"question": question},
        config={
            "configurable": {
                "session_id": "srusti-session"
            }
        }
    )

    print("Answer:")
    print(response)

print("\n" + "=" * 60)
print("Task 2 conversation memory test completed")
print("=" * 60)
# ============================================================
# TASK 3 - LANGCHAIN AGENT WITH 2 TOOLS
# ============================================================

from langchain_core.tools import tool
from langchain.agents import create_agent

# Tool 1: Web Search Stub
@tool
def web_search(query: str) -> str:
    """A simple web search stub that returns sample search information."""
    return (
        f"Web search results for '{query}': "
        "LangChain is a framework for building applications "
        "powered by language models."
    )


# Tool 2: Calculator
@tool
def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression."""
    try:
        # Allow only basic mathematical characters
        allowed = "0123456789+-*/(). "
        if not all(char in allowed for char in expression):
            return "Invalid mathematical expression."

        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)

    except Exception:
        return "Unable to calculate the expression."


# Create agent
agent = create_agent(
    model=llm,
    tools=[web_search, calculator],
    system_prompt=(
        "You are a helpful assistant. "
        "Use the calculator tool for mathematical calculations. "
        "Use the web_search tool when the user asks for information "
        "that requires a web search."
    )
)

print("\n" + "=" * 60)
print("       W6D2 - LANGCHAIN AGENT TEST")
print("=" * 60)

# Three agent tasks
agent_tasks = [
    "Calculate 125 * 8.",
    "Search for information about LangChain.",
    "Calculate (250 + 150) / 4."
]

for i, task in enumerate(agent_tasks, 1):

    print(f"\nAgent Task {i}: {task}")

    result = agent.invoke({
        "messages": [
            ("user", task)
        ]
    })

    # Display final agent response
    final_message = result["messages"][-1]

    print("Agent Answer:")
    print(final_message.content)


print("\n" + "=" * 60)
print("Task 3 completed: Agent tested with 2 tools and 3 tasks")
print("=" * 60)