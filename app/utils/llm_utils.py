from groq import Groq

def generate_answer(query, context_docs, groq_api_key, llm_model):
    """
    Generates an answer to a user query using provided loan context and an LLM model.

    Args:
        query (str): The user’s question.
        context_docs (list): A list of documents containing loan-related context.
        groq_api_key (str): API key for Groq service.
        llm_model (str): The language model to use for generating the answer.

    Returns:
        str: A concise, context-based answer to the user's query.

    """
    context_text = "\n\n".join([f"[Doc {i+1}] {doc['text']}" for i, doc in enumerate(context_docs)])

    prompt = f"""

You are a helpful assistant called 'Loan Product Assistant' for Bank of Maharashtra.

Rules:
1. Answer user questions ONLY using the information provided in the context.
2. If the answer is not found in the context, reply: "I could not find this information in the available documents."
3. Keep answers clear and concise.
4. Use bullet points or short paragraphs where possible for readability.
5. Do NOT add extra commentary like “let me know if you need more details.”

User Question:
{query}

Context from Bank of Maharashtra loan documents:
{context_text}

Answer the question in a clear, concise manner.

"""

    client = Groq(api_key=groq_api_key)

    completion = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a financial assistant specialized in Bank of Maharashtra loans."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
    )

    return completion.choices[0].message.content
