from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
from typing import List, Optional, AsyncGenerator
import json
import asyncio

router = APIRouter()
genai.configure(api_key="AIzaSyCWiZmhjdh1GmYKnvJMLvgsY-bh20wYOZs")  # Replace with your actual API key

class ChatMessage(BaseModel):
    """Model for chat message exchange"""
    message: str = Field(..., description="User's message/query")
    chat_history: Optional[List[dict]] = Field(
        None,
        description="Optional chat history for context in format [{'user': '...', 'bot': '...'}]"
    )

class ChatResponse(BaseModel):
    """Response model for agricultural chatbot"""
    response: str
    disclaimer: str = "AI suggestions should be verified with local agricultural experts"

class StreamingChatResponse(BaseModel):
    """Streaming response chunk model"""
    chunk: str
    is_complete: bool = False
    disclaimer: Optional[str] = None

# Non-streaming endpoint (original)
@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Agricultural Chatbot",
    description="""An AI-powered chatbot that answers general agricultural questions.
    Provide chat history for contextual conversations.""",
    tags=["Agricultural Chatbot"]
)
async def agricultural_chat(message: ChatMessage):
    """
    Chat with an agricultural expert AI about farming practices, crop issues, etc.
    
    - **message**: User's question or message
    - **chat_history**: Optional previous conversation for context
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        system_prompt = """You are an agricultural expert assistant with 20 years of field experience.
        Provide helpful, accurate information about:
        - Crop cultivation
        - Pest and disease management
        - Soil health
        - Irrigation techniques
        - Organic farming
        - Agricultural technology
        
        Response guidelines:
        1. Be concise but thorough
        2. Use simple language (8th grade level)
        3. Provide actionable advice
        4. Mention regional variations when relevant
        5. Always remind users to verify with local experts
        6. Format lists clearly with bullet points
        7. If unsure, say "I recommend consulting with your local agricultural extension office about..."
        """
        
        chat = model.start_chat(history=[])
        await chat.send_message_async(system_prompt)
        
        if message.chat_history:
            for turn in message.chat_history:
                if turn.get("user"):
                    await chat.send_message_async(turn["user"])
                if turn.get("bot"):
                    await chat.send_message_async(f"[Previous response]: {turn['bot']}")
        
        response = await chat.send_message_async(message.message)
        
        return ChatResponse(
            response=response.text,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate chat response: {str(e)}"
        )

# Streaming endpoint
@router.post(
    "/chat/stream",
    summary="Agricultural Chatbot (Streaming)",
    description="""An AI-powered chatbot that streams responses in real-time.
    Returns Server-Sent Events (SSE) for real-time response streaming.""",
    tags=["Agricultural Chatbot"]
)
async def agricultural_chat_stream(message: ChatMessage):
    """
    Stream chat responses from agricultural expert AI in real-time.
    
    - **message**: User's question or message
    - **chat_history**: Optional previous conversation for context
    
    Returns: Server-Sent Events (SSE) stream
    """
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Configure model for streaming
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            system_prompt = """You are an agricultural expert assistant with 20 years of field experience.
            Provide helpful, accurate information about:
            - Crop cultivation
            - Pest and disease management
            - Soil health
            - Irrigation techniques
            - Organic farming
            - Agricultural technology
            
            Response guidelines:
            1. Be concise but thorough
            2. Use simple language (8th grade level)
            3. Provide actionable advice
            4. Mention regional variations when relevant
            5. Always remind users to verify with local experts
            6. Format lists clearly with bullet points
            7. If unsure, say "I recommend consulting with your local agricultural extension office about..."
            """
            
            # Start chat session
            chat = model.start_chat(history=[])
            await chat.send_message_async(system_prompt)
            
            # Add chat history if provided
            if message.chat_history:
                for turn in message.chat_history:
                    if turn.get("user"):
                        await chat.send_message_async(turn["user"])
                    if turn.get("bot"):
                        await chat.send_message_async(f"[Previous response]: {turn['bot']}")
            
            # Send message with streaming enabled
            response = chat.send_message(message.message, stream=True)
            
            # Stream the response chunks
            for chunk in response:
                if chunk.text:
                    chunk_data = {
                        "chunk": chunk.text,
                        "is_complete": False
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)
            
            # Send completion signal with disclaimer
            final_chunk = {
                "chunk": "",
                "is_complete": True,
                "disclaimer": "AI suggestions should be verified with local agricultural experts"
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
        except Exception as e:
            # Send error as final chunk
            error_chunk = {
                "chunk": f"Error: {str(e)}",
                "is_complete": True,
                "error": True
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

# Alternative: JSON Lines streaming endpoint
@router.post(
    "/chat/stream-jsonl",
    summary="Agricultural Chatbot (JSON Lines Streaming)",
    description="""An AI-powered chatbot that streams responses as JSON Lines.
    Each line is a JSON object with response chunks.""",
    tags=["Agricultural Chatbot"]
)
async def agricultural_chat_stream_jsonl(message: ChatMessage):
    """
    Stream chat responses as JSON Lines format.
    
    - **message**: User's question or message
    - **chat_history**: Optional previous conversation for context
    
    Returns: JSON Lines stream (one JSON object per line)
    """
    
    async def generate_jsonl_stream() -> AsyncGenerator[str, None]:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            system_prompt = """You are an agricultural expert assistant with 20 years of field experience.
            Provide helpful, accurate information about:
            - Crop cultivation
            - Pest and disease management
            - Soil health
            - Irrigation techniques
            - Organic farming
            - Agricultural technology
            
            Response guidelines:
            1. Be concise but thorough
            2. Use simple language (8th grade level)
            3. Provide actionable advice
            4. Mention regional variations when relevant
            5. Always remind users to verify with local experts
            6. Format lists clearly with bullet points
            7. If unsure, say "I recommend consulting with your local agricultural extension office about..."
            """
            
            chat = model.start_chat(history=[])
            await chat.send_message_async(system_prompt)
            
            if message.chat_history:
                for turn in message.chat_history:
                    if turn.get("user"):
                        await chat.send_message_async(turn["user"])
                    if turn.get("bot"):
                        await chat.send_message_async(f"[Previous response]: {turn['bot']}")
            
            # Send message with streaming
            response = chat.send_message(message.message, stream=True)
            
            # Stream chunks as JSON Lines
            for chunk in response:
                if chunk.text:
                    chunk_data = StreamingChatResponse(
                        chunk=chunk.text,
                        is_complete=False
                    )
                    yield f"{chunk_data.model_dump_json()}\n"
                    await asyncio.sleep(0.01)
            
            # Final chunk with disclaimer
            final_chunk = StreamingChatResponse(
                chunk="",
                is_complete=True,
                disclaimer="AI suggestions should be verified with local agricultural experts"
            )
            yield f"{final_chunk.model_dump_json()}\n"
            
        except Exception as e:
            error_chunk = StreamingChatResponse(
                chunk=f"Error: {str(e)}",
                is_complete=True
            )
            yield f"{error_chunk.model_dump_json()}\n"
    
    return StreamingResponse(
        generate_jsonl_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )