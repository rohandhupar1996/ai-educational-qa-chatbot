"""
LLM Model Integration - OpenAI API Handler (Fixed)
"""
import os
import logging
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

class QuestionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # Fix pydantic warning
    
    question: str
    max_length: int = 4
    context: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    success: bool
    error: Optional[str] = None

class VoiceCloneRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # Fix pydantic warning
    
    question: str
    voice_id: str
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.7

class OpenAIModel:
    """
    OpenAI model handler for generating responses to AI-related questions.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI model.
        
        Args:
            api_key: OpenAI API key (if not provided, will look for OPENAI_API_KEY env var)
            model_name: Name of the OpenAI model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client - simplified to avoid version conflicts
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            # Fallback for older versions
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=30.0
            )
        
        logger.info(f"Initialized OpenAI model: {self.model_name}")
    
    def generate_answer(self, question: str, max_lines: int = 4, context: Optional[str] = None) -> QuestionResponse:
        """
        Generate an answer to a question using OpenAI.
        
        Args:
            question: The question to answer
            max_lines: Maximum number of lines in the response
            context: Additional context for the question
            
        Returns:
            QuestionResponse object with answer and status
        """
        try:
            if not question.strip():
                return QuestionResponse(
                    answer="",
                    success=False,
                    error="Question cannot be empty"
                )
            
            # Build messages for chat completion
            messages = []
            
            if context:
                system_message = f"You are an AI assistant specializing in Artificial Intelligence topics. Context: {context}. Provide clear, informative answers in {max_lines} lines maximum."
            else:
                system_message = f"You are an AI assistant specializing in Artificial Intelligence topics. Provide clear, informative answers about AI in {max_lines} lines maximum. Be educational yet concise."
            
            messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": question})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=min(max_lines * 50, 500),
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            if response and response.choices and len(response.choices) > 0:
                answer = response.choices[0].message.content.strip()
                logger.info(f"Generated answer for question: {question[:50]}...")
                
                return QuestionResponse(
                    answer=answer,
                    success=True
                )
            else:
                return QuestionResponse(
                    answer="",
                    success=False,
                    error="No response generated from OpenAI"
                )
                
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            return QuestionResponse(
                answer="",
                success=False,
                error=f"Error generating response: {str(e)}"
            )