# main.py

from graph import graph
from langchain_core.messages import HumanMessage


result = graph.invoke({
    "messages": [
        HumanMessage(
            content="I want to register. My name is Ramakrishna G and my WhatsApp number is 919640864111"
        )
    ],

    "next_agent": None,

    "client_name": None,
    "primary_no": None,
    "secondary_no": None,

    "mobile": None,
    "store_id": None,
    "bookdate": None,
    "booktime": None,

    "api_result": None,
    "completed": False
})


for message in result["messages"]:
    print(message)