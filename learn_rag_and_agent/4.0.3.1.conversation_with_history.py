from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# 1. 커스텀 대화 체인 클래스 정의
class MyConversationChain(Runnable):
    """
    대화 기록 로드, 체인 실행, 대화 기록 저장을 통합한 커스텀 Runnable 클래스.
    """
    def __init__(self, llm, prompt, memory, input_key="input"):
        self.prompt = prompt
        self.memory = memory
        self.input_key = input_key

        # LCEL을 사용하여 체인 파이프라인 정의
        self.chain = (
            # 1단계: 메모리 로드 및 입력에 'chat_history' 추가
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                | itemgetter(memory.memory_key)
            )
            # 2단계: 프롬프트 구성 및 LLM 호출
            | prompt
            | llm
            # 3단계: 응답을 문자열로 파싱
            | StrOutputParser()
        )

    def invoke(self, query, configs=None, **kwargs):
        # 1. 체인 실행 (메모리를 포함하여 응답 생성)
        answer = self.chain.invoke({self.input_key: query})
        
        # 2. 대화 기록 저장 (다음 호출을 위해 현재 대화 내용을 메모리에 업데이트)
        self.memory.save_context(inputs={"human": query}, outputs={"ai": answer})
        
        # 3. 응답 반환
        return answer

# --- 초기 설정 ---

# ChatOpenAI 모델 초기화
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# 대화형 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 대화 버퍼 메모리 생성 (대화 전문 저장)
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# 요약 메모리 옵션 (대화가 길어질 경우 사용):
# memory = ConversationSummaryMemory(
#     llm=llm, return_messages=True, memory_key="chat_history"
# )

# --- 커스텀 대화 체인 사용 ---

conversation_chain = MyConversationChain(llm, prompt, memory)

# 1. 첫 번째 대화 (이름 소개)
print(f"Human: 안녕하세요? 만나서 반갑습니다. 제 이름은 테디 입니다.")
response1 = conversation_chain.invoke("안녕하세요? 만나서 반갑습니다. 제 이름은 테디 입니다.")
print(f"AI: {response1}\n")

# 2. 두 번째 대화 (이름 기억 확인) - 1차 대화 내용 참조
print(f"Human: 제 이름이 뭐라고요?")
response2 = conversation_chain.invoke("제 이름이 뭐라고요?")
print(f"AI: {response2}\n")

# 3. 세 번째 대화 (응답 언어 설정) - 1, 2차 대화 내용 참조
print(f"Human: 앞으로는 영어로만 답변해주세요 알겠어요?")
response3 = conversation_chain.invoke("앞으로는 영어로만 답변해주세요 알겠어요?")
print(f"AI: {response3}\n")

# 4. 네 번째 대화 (이름 확인 + 영어 답변) - 1, 2, 3차 대화 내용 모두 참조
print(f"Human: 제 이름을 다시 한 번 말해주세요")
response4 = conversation_chain.invoke("제 이름을 다시 한 번 말해주세요")
print(f"AI: {response4}\n")

# --- 최종 메모리 상태 확인 ---
print("-" * 30)
print("최종 메모리 상태 확인:")

final_history = conversation_chain.memory.load_memory_variables({})["chat_history"]

for message in final_history:
    print(f"[{message.type.capitalize()}]: {message.content}")