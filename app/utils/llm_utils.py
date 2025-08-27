from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_answer(query, docs, groq_api_key, llm_model):
    """
    Uses LangChain ChatGroq with a custom prompt.
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model=llm_model,
        temperature=0.2,
        max_tokens=512
    )

    context_text = "\n\n".join([f"[Doc {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

    prompt_template = """
You are a helpful assistant called 'Loan Product Assistant' for Bank of Maharashtra.

Rules:
1. Answer user questions ONLY using the information provided in the context.
2. If the answer is not found in the context, reply: "I could not find this information in the available documents."
3. Keep answers clear and concise.
4. Use bullet points or short paragraphs where possible for readability.

User Question:
{question}

Context:
{context}

Answer:
    """
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"question": query, "context": context_text})
