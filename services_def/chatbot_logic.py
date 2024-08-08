# chatbot_logic.py

from typing import TypedDict
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_upstage import UpstageGroundednessCheck
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from operator import itemgetter
import logging
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    relevance: str


def setup_chatbot(content):
    try:
        logger.info("Setting up chatbot")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50
        )
        documents = [Document(page_content=content)]
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split content into {len(split_docs)} documents")

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(split_docs, embeddings)
        retriever = db.as_retriever()
        logger.info("Created FAISS index and retriever")

        prompt = hub.pull("teddynote/rag-korean-with-source")
        chain = (
            {"question": itemgetter("question"), "context": itemgetter("context")}
            | prompt
            | ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
            | StrOutputParser()
        )
        logger.info("Created LangChain components")

        upstage_ground_checker = UpstageGroundednessCheck()

        def retrieve_document(state: GraphState) -> GraphState:
            retrieved_docs = retriever.invoke(state["question"])
            # retrieved_docs가 Document 객체의 리스트인지 확인
            if isinstance(retrieved_docs, list) and all(
                isinstance(doc, Document) for doc in retrieved_docs
            ):
                return GraphState(context=retrieved_docs)
            else:
                # 필요한 경우 Document 객체로 변환
                context = [
                    Document(page_content=doc) if isinstance(doc, str) else doc
                    for doc in retrieved_docs
                ]
                return GraphState(context=context)

        def llm_answer(state: GraphState) -> GraphState:
            question = state["question"]
            context = state["context"]
            response = chain.invoke({"question": question, "context": context})
            return GraphState(answer=response)

        def rewrite(state):
            question = state["question"]
            answer = state["answer"]
            context = state["context"]
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a professional prompt rewriter. Your task is to generate the question in order to get additional information that is now shown in the context. Your generated question will be searched on the web to find relevant information.",
                    ),
                    (
                        "human",
                        "Rewrite the question to get additional information to get the answer.\n\nHere is the initial question:\n-------\n{question}\n-------\n\nHere is the initial context:\n-------\n{context}\n-------\n\nHere is the initial answer to the question:\n-------\n{answer}\n-------\n\nFormulate an improved question in Korean:",
                    ),
                ]
            )
            model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
            chain = prompt | model | StrOutputParser()
            response = chain.invoke(
                {"question": question, "answer": answer, "context": context}
            )
            return GraphState(question=response)

        def search_on_web(state: GraphState) -> GraphState:
            search_tool = TavilySearchResults(max_results=5)
            search_result = search_tool.invoke({"query": state["question"]})
            return GraphState(context=search_result)

        def relevance_check(state: GraphState) -> GraphState:
            context = state["context"]
            answer = state["answer"]

            # context가 Document 객체의 리스트인 경우 텍스트로 변환
            if isinstance(context, list) and all(
                isinstance(doc, Document) for doc in context
            ):
                context_text = "\n".join(doc.page_content for doc in context)
            else:
                context_text = str(context)  # 다른 타입의 경우 문자열로 변환

            response = upstage_ground_checker.run(
                {"context": context_text, "answer": answer}
            )
            return GraphState(
                relevance=response, question=state["question"], answer=answer
            )

        def is_relevant(state: GraphState) -> GraphState:
            return state["relevance"]

        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve_document)
        workflow.add_node("llm_answer", llm_answer)
        workflow.add_node("relevance_check", relevance_check)
        workflow.add_node("rewrite", rewrite)
        workflow.add_node("search_on_web", search_on_web)

        workflow.add_edge("retrieve", "llm_answer")
        workflow.add_edge("llm_answer", "relevance_check")
        workflow.add_edge("rewrite", "search_on_web")
        workflow.add_edge("search_on_web", "llm_answer")

        workflow.add_conditional_edges(
            "relevance_check",
            is_relevant,
            {
                "grounded": END,
                "notGrounded": "rewrite",
                "notSure": "rewrite",
            },
        )

        workflow.set_entry_point("retrieve")
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        logger.info("Workflow compiled successfully")

        return app
    except Exception as e:
        logger.error(f"Error in setup_chatbot: {str(e)}", exc_info=True)
        raise


def generate_response(app, user_input):
    try:
        logger.info(f"Generating response for input: {user_input}")
        config = RunnableConfig(
            recursion_limit=12, configurable={"thread_id": "CORRECTIVE-SEARCH-RAG"}
        )
        inputs = GraphState(question=user_input)
        result = app.invoke(inputs, config=config)
        logger.info("Response generated successfully")
        return result["answer"]
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
        raise


# 유틸리티 함수들 (format_docs, format_searched_docs 등)은 여기에 추가해야 합니다.
