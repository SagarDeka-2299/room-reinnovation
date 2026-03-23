import os
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Literal, List, cast
from langgraph.graph import StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class State(BaseModel):
    story:str=''
    need_reiteration:bool=True
    chat_memory:List[SystemMessage|HumanMessage|AIMessage]=[
        SystemMessage(content="You are horror story writer, you think of new bezar ideas and write unique short horror story"),
        HumanMessage(content="Write a horror story, indian village style")
    ]

class AI2HumanRequest(BaseModel):
    req:str="Give feedback on the story I just wrote"
    data:str=''
class Human2AIRequest(BaseModel):
    approved:bool
    msg:str=''

def genStory(state:State)->State:
    llm=ChatGoogleGenerativeAI(model='gemini-3-flash-preview', temperature=0.9,google_api_key=os.getenv("GOOGLE_FREE_API_KEY"))
    story_response=llm.invoke(state.chat_memory)
    state.story=cast(str,story_response.content)
    return state

def getFeedback(state:State)->State:
    feedback=cast(Human2AIRequest, interrupt(AI2HumanRequest(data=state.story)))
    state.need_reiteration=not feedback.approved
    return state

def routeEdit(state:State)->Literal['poet','__end__']:
    return 'poet' if state.need_reiteration else '__end__'

graph=StateGraph(State)

graph.add_node('poet', genStory)
graph.add_node("feedback", getFeedback)

graph.add_edge(START, 'poet')
graph.add_edge('poet','feedback')
graph.add_conditional_edges('feedback',routeEdit)


ckptr=MemorySaver()
workflow=graph.compile(checkpointer=ckptr)
test_output=workflow.invoke(State(), config={'configurable':{'thread_id':'temp'}})
