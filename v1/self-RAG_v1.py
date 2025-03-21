"""
重構後的 Self-RAG 程式（已移除 web_search 邏輯）
------------------------------------------------
此版本將文件載入、向量庫建立、各個 LLM 鏈（router、retrieval grader、生成、問句重寫等）
以及流程圖中的節點函式拆分為獨立函式，且僅使用 vectorstore 進行檢索。
"""

from utils import load_and_prompt_env
from typing import List, Dict, Any, TypedDict, Literal

# LangChain 與相關元件
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# 載入環境變數
load_and_prompt_env(["OPENAI_API_KEY"])

#############################
# 1. 向量庫與文件處理模組
#############################
def build_vectorstore(urls: List[str], chunk_size: int = 500, chunk_overlap: int = 0) -> Any:
    """從 URL 載入文件，切分文本並建立向量庫（Chroma）"""
    # 初始化 embeddings
    embd = OpenAIEmbeddings()

    # 載入文件
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # 文本切分
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # 建立向量庫
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embd,
    )
    return vectorstore

#############################
# 2. 定義資料模型與 LLM 鏈
#############################

# Router 模型與鏈（此處僅使用 vectorstore，不再有 web search 選項）
class RouteQuery(BaseModel):
    """依據問題路由到 vectorstore"""
    datasource: Literal["vectorstore"] = Field(
        ...,
        description="決定使用向量庫進行查詢",
    )

def build_router_chain() -> Any:
    system = (
        "You are an expert at routing a user question to a vectorstore.\n"
        "The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks."
    )
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])
    llm = ChatOpenAI(model="gpt-4o")
    structured_llm_router = llm.with_structured_output(RouteQuery)
    # 管道串接
    return route_prompt | structured_llm_router

# Retrieval Grader 模型與鏈
class GradeDocuments(BaseModel):
    """檢查檔案是否與問題相關，回傳 binary score"""
    binary_score: str = Field(
        description="若文件與問題相關，請回傳 'yes'，否則 'no'"
    )

def build_retrieval_grader_chain() -> Any:
    system = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
        "It does not need to be a stringent test. The goal is to filter out erroneous retrievals.\n"
        "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."
    )
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    return grade_prompt | structured_llm_grader

# RAG 生成鏈：使用 hub 上的 prompt 與 LLM 生成答案
def build_rag_chain() -> Any:
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return prompt | llm | StrOutputParser()

# Hallucination Grader 模型與鏈
class GradeHallucinations(BaseModel):
    """檢查生成答案是否脫離文件內容"""
    binary_score: str = Field(
        description="如果答案是根據文件生成，請回傳 'yes'；否則 'no'"
    )

def build_hallucination_grader_chain() -> Any:
    system = (
        "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n"
        "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."
    )
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    return hallucination_prompt | structured_llm_grader

# Answer Grader 模型與鏈
class GradeAnswer(BaseModel):
    """檢查生成答案是否有解決問題"""
    binary_score: str = Field(
        description="如果答案有解決問題，請回傳 'yes'；否則 'no'"
    )

def build_answer_grader_chain() -> Any:
    system = (
        "You are a grader assessing whether an answer addresses / resolves a question.\n"
        "Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."
    )
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)
    return answer_prompt | structured_llm_grader

# Question Rewriter 鏈
def build_question_rewriter_chain() -> Any:
    system = (
        "You are a question re-writer that converts an input question to a better version that is optimized\n"
        "for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."
    )
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    return re_write_prompt | llm | StrOutputParser()

#############################
# 3. Graph 節點函式
#############################
# 定義 Graph State 型別
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Any]

def retrieve(state: Dict) -> Dict:
    """從向量庫中檢索文件"""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def generate(state: Dict) -> Dict:
    """使用 RAG 生成答案"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state: Dict) -> Dict:
    """根據問題篩選出相關文件"""
    print("---GRADE DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score.lower() == "yes":
            print("---DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---DOCUMENT NOT RELEVANT---")
    return {"documents": filtered_docs, "question": question}

def transform_query(state: Dict) -> Dict:
    """將問題重寫以優化檢索效果"""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": state["documents"], "question": better_question}

def route_question(state: Dict) -> str:
    """依據問題內容路由，目前只使用 vectorstore"""
    print("---ROUTE QUESTION---")
    # 無論 LLM 回傳何種結果，皆直接使用向量庫檢索
    return "retrieve"

def decide_to_generate(state: Dict) -> str:
    """判斷是否有足夠相關文件，決定接下來生成答案或重寫問題"""
    print("---DECIDE NEXT STEP BASED ON DOCUMENT RELEVANCE---")
    if not state["documents"]:
        print("---NO RELEVANT DOCUMENTS, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---RELEVANT DOCUMENTS FOUND, GENERATE ANSWER---")
        return "generate"

def grade_generation_v_documents_and_question(state: Dict) -> str:
    """檢查生成答案是否基於檢索文件且回答問題"""
    print("---GRADE GENERATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    if score.binary_score.lower() == "yes":
        print("---GENERATION IS GROUNDED---")
        score_ans = answer_grader.invoke({"question": question, "generation": generation})
        if score_ans.binary_score.lower() == "yes":
            print("---GENERATION ADDRESSES THE QUESTION---")
            return "useful"
        else:
            print("---GENERATION DOES NOT ADDRESS THE QUESTION---")
            return "not useful"
    else:
        print("---GENERATION IS NOT GROUNDED, RETRY---")
        return "not supported"

#############################
# 4. 主程式與 Graph 編排
#############################
from langgraph.graph import END, StateGraph, START

def build_and_compile_graph() -> Any:
    workflow = StateGraph(GraphState)

    # 定義各節點
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # 建立流程邊界：由於僅使用向量庫，直接從 START 進入 retrieve 節點
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"}
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {"not supported": "generate", "useful": END, "not useful": "transform_query"}
    )

    return workflow.compile()

def run_graph(app: Any, question: str) -> None:
    """執行 Graph 並列印各節點狀態與最終生成答案"""
    from pprint import pprint
    print(f"\n=== 執行問題：{question} ===")
    inputs = {"question": question}
    for output in app.stream(inputs):
        for node, state in output.items():
            pprint(f"Node '{node}': state keys -> {list(state.keys())}")
        print("\n---\n")
    # 印出最終生成答案
    pprint(f"Final Generation: {state.get('generation', 'No generation')}")

if __name__ == "__main__":
    # 建立向量庫與檢索器
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    vectorstore = build_vectorstore(urls)
    retriever = vectorstore.as_retriever()

    # 建立各個 LLM 鏈
    question_router = build_router_chain()
    retrieval_grader = build_retrieval_grader_chain()
    rag_chain = build_rag_chain()
    hallucination_grader = build_hallucination_grader_chain()
    answer_grader = build_answer_grader_chain()
    question_rewriter = build_question_rewriter_chain()

    # 編排 Graph
    app = build_and_compile_graph()

    # 若在 Jupyter 環境中，可視化流程圖
    try:
        from IPython.display import Image, display
        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception:
        pass

    # 執行 Graph 測試不同問題
    run_graph(app, "What player at the Bears expected to draft first in the 2024 NFL draft?")
    run_graph(app, "What are the types of agent memory?")

    from IPython.display import Image, display

    display(Image(app.get_graph().draw_mermaid_png()))
