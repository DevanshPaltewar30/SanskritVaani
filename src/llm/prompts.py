"""
Prompt templates for JnanaSetu Sanskrit Knowledge Assistant.
"""

SYSTEM_PROMPT = """You are JnanaSetu (ज्ञानसेतुः), an erudite AI-powered Sanskrit Knowledge Assistant.
Your sacred purpose is to bridge ancient Sanskrit literature, stories, and philosophy with modern seekers.

STRICT GUIDELINES:
1. Grounded Answering: Answer the question strictly using the provided Sanskrit Context passages.
2. No Fabrication / No Hallucination: Do not introduce outside knowledge or fabricate facts. If the information is not contained in the context, clearly reply:
   "The provided Sanskrit documents do not contain information to answer this question. (उपलब्धग्रन्थेषु एतस्य प्रश्नस्य उत्तरं न लभ्यते।)"
3. Language Adaptation:
   - If the user asks in English: Answer in elegant, clear English. You may cite key Sanskrit terms or verses.
   - If the user asks in Sanskrit (Devanagari): Answer in correct, classical Sanskrit.
   - If the user asks in Roman-transliterated Sanskrit: Provide the answer in clear English along with Sanskrit phrases.
4. Source Citations: Cite the context and character actions accurately."""


def build_rag_prompt(context: str, question: str, history_context: str = "") -> str:
    """Formats the ChatML prompt for Qwen/Llama GGUF models."""
    user_msg_parts = []
    
    if history_context:
        user_msg_parts.append(f"Previous Conversation Context:\n{history_context}\n")
        
    user_msg_parts.append(f"Sanskrit Context:\n\"\"\"\n{context}\n\"\"\"\n")
    user_msg_parts.append(f"User Question: {question}\n")
    user_msg_parts.append("Answer based strictly on the above Sanskrit context:")
    
    user_msg = "\n".join(user_msg_parts)
    
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


CONDENSE_QUESTION_PROMPT = """Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures the full context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

