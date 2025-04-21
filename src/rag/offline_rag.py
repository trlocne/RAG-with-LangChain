import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(self,
                       text_response: str,
                       pattern: str = r"Answer\s*:\s*(.*)") -> str:
        # Dùng re.DOTALL để match toàn bộ nội dung kể cả xuống dòng
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text.split("\n")[0].strip()
        else:
            return text_response.strip()

class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = hub.pull("rlm/rag-prompt")
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
        }

        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | RunnableLambda(self.str_parser.parse)
        )

        return rag_chain

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
