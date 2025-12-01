"""
Conversation History 예시 코드
LangChain을 사용하여 대화 기록을 관리하는 챗봇 구현
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# API KEY 정보 로드
load_dotenv()


# 세션 기록을 저장할 딕셔너리
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    세션 ID를 기반으로 세션 기록을 가져오는 함수
    
    Args:
        session_id: 세션을 구분하는 고유 ID
        
    Returns:
        BaseChatMessageHistory: 해당 세션의 대화 기록
    """
    print(f"[대화 세션ID]: {session_id}")
    if session_id not in store:
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def create_chain_with_history():
    """
    대화 기록을 포함한 체인을 생성하는 함수
    
    Returns:
        RunnableWithMessageHistory: 대화 기록이 포함된 체인
    """
    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),
        ]
    )

    # LLM 생성
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    # 일반 Chain 생성
    chain = prompt | llm | StrOutputParser()

    # 대화 기록을 포함한 체인 생성
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history


def chat_session_example():
    """
    대화 세션 예시를 실행하는 함수
    """
    # 체인 생성
    chain = create_chain_with_history()
    
    # 세션 ID 설정
    session_id = "user_001"
    
    print("=" * 60)
    print("대화 세션 시작")
    print("=" * 60)
    
    # 첫 번째 질문
    print("\n[질문 1]")
    response1 = chain.invoke(
        {"question": "나의 이름은 테디입니다."},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"답변: {response1}")
    
    # 두 번째 질문 (이전 대화 기록 참조)
    print("\n[질문 2]")
    response2 = chain.invoke(
        {"question": "내 이름이 뭐라고?"},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"답변: {response2}")
    
    # 세 번째 질문 (이전 대화 기록 참조)
    print("\n[질문 3]")
    response3 = chain.invoke(
        {"question": "나에 대해 알고 있는 정보를 요약해줘."},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"답변: {response3}")
    
    print("\n" + "=" * 60)
    print("새로운 세션 시작 (다른 session_id)")
    print("=" * 60)
    
    # 다른 세션 ID로 질문 (새로운 대화)
    print("\n[질문 4 - 새 세션]")
    response4 = chain.invoke(
        {"question": "내 이름이 뭐라고?"},
        config={"configurable": {"session_id": "user_002"}},
    )
    print(f"답변: {response4}")


def interactive_chat():
    """
    대화형 챗봇 실행 함수
    """
    chain = create_chain_with_history()
    session_id = "interactive_session"
    
    print("=" * 60)
    print("대화형 챗봇 시작 (종료하려면 'quit' 또는 'exit' 입력)")
    print("=" * 60)
    
    while True:
        user_input = input("\n질문: ")
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("챗봇을 종료합니다.")
            break
            
        if not user_input.strip():
            continue
            
        try:
            response = chain.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            print(f"답변: {response}")
        except Exception as e:
            print(f"오류 발생: {e}")


def view_session_history(session_id: str):
    """
    특정 세션의 대화 기록을 출력하는 함수
    
    Args:
        session_id: 조회할 세션 ID
    """
    if session_id in store:
        print(f"\n[세션 {session_id}의 대화 기록]")
        history = store[session_id]
        messages = history.messages
        
        for i, msg in enumerate(messages, 1):
            print(f"{i}. {msg.type}: {msg.content}")
    else:
        print(f"세션 {session_id}의 기록이 없습니다.")


if __name__ == "__main__":
    # 예시 1: 미리 정의된 대화 세션 실행
    print("\n### 예시 1: 미리 정의된 대화 세션 ###")
    chat_session_example()
    
    # 대화 기록 조회
    print("\n\n### 대화 기록 조회 ###")
    view_session_history("user_001")
    view_session_history("user_002")