#!/usr/bin/env python
# coding: utf-8

# # Chatbot Program
# 
# #### Chatbot with Evaluator - Hugging Face Deployment Ready
# - Primary Agent: Google Gemini (via OpenAI API)
# - Evaluator: Groq Llama 3.3 70B
# - Fast API-based inference (no local models)

# In[ ]:


# imports

import os
import gradio as gr
from openai import OpenAI
import time
from typing import Tuple, Optional
import json
from dotenv import load_dotenv


# In[ ]:


load_dotenv(override=True)


# In[ ]:


# Check for API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GOOGLE_API_KEY:
    print(f"Google API Key exists and begins {GOOGLE_API_KEY[:2]}")
else:
    print("Google API Key not set (and this is optional)")

if GROQ_API_KEY:
    print(f"Groq API Key exists and begins {GROQ_API_KEY[:4]}")
else:
    print("Groq API Key not set (and this is optional)")


# In[ ]:


# Model configurations
AGENT_MODELS = {
    # "Gemini Pro": {
    #     "model": "gemini-pro",
    #     "description": "Google's Gemini Pro model",
    #     "max_tokens": 2048
    # },
    "Gemini 1.5 flash": {
        "model": "gemini-1.5-flash", 
        "description": "Fast Gemini model",
        "max_tokens": 2048
    }
    # "Gemini 1.5 Pro": {
    #     "model": "gemini-1.5-pro",
    #     "description": "Advanced Gemini model",
    #     "max_tokens": 2048
    # }
}

EVALUATOR_MODELS = {
    "Llama 3.3 70B": {
        "model": "llama-3.3-70b-versatile",
        "description": "Groq's Llama 3.3 70B - Fast & Powerful"
    }
    # "Llama 3.1 70B": {
    #     "model": "llama-3.1-70b-versatile",
    #     "description": "Groq's Llama 3.1 70B"
    # },
    # "Mixtral 8x7B": {
    #     "model": "mixtral-8x7b-32768",
    #     "description": "Groq's Mixtral model"
    # }
}


# In[ ]:


# ===========================
# API Client Management Class
# ===========================

class APIClientManager:
    def __init__(self):
        self.gemini_client = None
        self.groq_client = None
        self.errors = []
        self.initialize_clients()
    
    def initialize_clients(self):
        """Initialize API clients with error handling."""
        # Get API keys from environment
        google_api_key = os.getenv("GOOGLE_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Initialize Gemini client
        if google_api_key:
            try:
                self.gemini_client = OpenAI(
                    api_key=google_api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                print("✅ Gemini API client initialized")
            except Exception as e:
                self.errors.append(f"Gemini initialization error: {e}")
        else:
            self.errors.append("GOOGLE_API_KEY not found in environment variables")
        
        # Initialize Groq client
        if groq_api_key:
            try:
                self.groq_client = OpenAI(
                    api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                print("✅ Groq API client initialized")
            except Exception as e:
                self.errors.append(f"Groq initialization error: {e}")
        else:
            self.errors.append("GROQ_API_KEY not found in environment variables")
    
    def create_evaluator_prompt(self, user_input: str, agent_response: str) -> str:
        """Create the evaluation prompt."""
        evaluator_prompt = (
            "You are an evaluator that decides whether a response to a question is acceptable. "
            "You are provided with a conversation between a User and an Agent. "
            "Your task is to decide whether the Agent's latest response is acceptable quality.\n\n"
            f"User Question: {user_input}\n\n"
            f"Agent Response: {agent_response}\n\n"
            "With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback.\n\n"
            "Format your evaluation as follows:\n"
            "1. Start with either 'ACCEPTABLE ✅' or 'UNACCEPTABLE ❌'\n"
            "2. Provide a brief quality score (1-10)\n"
            "3. List 2-3 specific strengths or issues\n"
            "4. Suggest one improvement if needed"
        )
        return evaluator_prompt
    
    def generate_agent_response(
        self,
        user_input: str,
        model_name: str = "Gemini 1.5 flash",
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Tuple[str, str, float]:
        """Generate response using Gemini API."""
        
        if not self.gemini_client:
            return "❌ Gemini API not initialized. Please set GOOGLE_API_KEY environment variable.", "Error", 0
        
        try:
            model_config = AGENT_MODELS.get(model_name, AGENT_MODELS["Gemini 1.5 flash"])
            model_id = model_config["model"]
            
            # Make API call to Gemini
            start_time = time.time()
            
            response = self.gemini_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."},
                    {"role": "user", "content": user_input}
                ],
                temperature=temperature,
                max_tokens=min(max_tokens, model_config["max_tokens"]),
                top_p=0.9
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract response
            agent_response = response.choices[0].message.content
            status = f"✅ {model_name} responded in {elapsed_time:.2f}s"
            
            return agent_response, status, elapsed_time
            
        except Exception as e:
            error_msg = f"❌ Gemini API error: {str(e)}"
            print(error_msg)
            
            # Check for common errors
            if "API key" in str(e):
                error_msg = "❌ Invalid Google API key. Please check GOOGLE_API_KEY."
            elif "quota" in str(e).lower():
                error_msg = "❌ API quota exceeded. Please try again later."
            elif "model" in str(e).lower():
                error_msg = f"❌ Model '{model_name}' not available. Try another model."
                
            return error_msg, "Error", 0
    
    def evaluate_response(
        self,
        user_input: str,
        agent_response: str,
        evaluator_model: str = "Llama 3.3 70B",
        temperature: float = 0.3
    ) -> Tuple[str, str, float]:
        """Evaluate the agent's response using Groq API."""
        
        if not self.groq_client:
            return "❌ Groq API not initialized. Please set GROQ_API_KEY environment variable.", "Error", 0
        
        try:
            model_config = EVALUATOR_MODELS.get(evaluator_model, EVALUATOR_MODELS["Llama 3.3 70B"])
            model_id = model_config["model"]
            
            # Create evaluation prompt using the class method
            eval_prompt = self.create_evaluator_prompt(user_input, agent_response)
            
            # Make API call to Groq
            start_time = time.time()
            
            response = self.groq_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a critical evaluator. Be honest but constructive in your feedback."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=temperature,
                max_tokens=300,
                top_p=0.9
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract evaluation
            evaluation = response.choices[0].message.content
            
            # Determine status based on evaluation
            if "ACCEPTABLE" in evaluation.upper():
                status = f"✅ Evaluation: Acceptable | {evaluator_model} ({elapsed_time:.2f}s)"
            elif "UNACCEPTABLE" in evaluation.upper():
                status = f"❌ Evaluation: Needs Improvement | {evaluator_model} ({elapsed_time:.2f}s)"
            else:
                status = f"🔍 Evaluation Complete | {evaluator_model} ({elapsed_time:.2f}s)"
            
            return evaluation, status, elapsed_time
            
        except Exception as e:
            error_msg = f"❌ Groq API error: {str(e)}"
            print(error_msg)
            
            # Check for common errors
            if "API key" in str(e):
                error_msg = "❌ Invalid Groq API key. Please check GROQ_API_KEY."
            elif "rate" in str(e).lower():
                error_msg = "❌ Rate limit exceeded. Please wait a moment and try again."
            elif "model" in str(e).lower():
                error_msg = f"❌ Model '{evaluator_model}' not available."
                
            return error_msg, "Error", 0


# In[ ]:


# ===========================
# Initialize Global Client Manager
# ===========================

api_manager = APIClientManager()


# In[ ]:


# ===========================
# Main Processing Function
# ===========================

def process_with_evaluation(
    user_input: str,
    agent_model: str,
    evaluator_model: str,
    temperature: float,
    max_tokens: int,
    enable_evaluation: bool
) -> Tuple[str, str, str, str]:
    """Process user input through agent and optionally evaluate."""
    
    if not user_input.strip():
        return "Please enter a message.", "", "No input provided", ""
    
    # Step 1: Generate agent response
    agent_response, agent_status, agent_time = api_manager.generate_agent_response(
        user_input,
        agent_model,
        temperature,
        max_tokens
    )
    
    # Step 2: Evaluate response (if enabled)
    if enable_evaluation and "Error" not in agent_status:
        evaluation, eval_status, eval_time = api_manager.evaluate_response(
            user_input,
            agent_response,
            evaluator_model,
            temperature=0.3  # Lower temp for evaluation
        )
        
        # Combine status
        total_time = agent_time + eval_time
        combined_status = f"Agent: {agent_model} ({agent_time:.2f}s) | Evaluator: {evaluator_model} ({eval_time:.2f}s) | Total: {total_time:.2f}s"
        
        # Format evaluation for better display
        if "ACCEPTABLE" in evaluation.upper():
            eval_summary = "✅ Response Quality: ACCEPTABLE"
        elif "UNACCEPTABLE" in evaluation.upper():
            eval_summary = "❌ Response Quality: NEEDS IMPROVEMENT"
        else:
            eval_summary = "🔍 Evaluation Complete"
            
    else:
        evaluation = "Evaluation disabled or skipped due to error" if not enable_evaluation else "Skipped due to agent error"
        eval_summary = "🔕 No evaluation performed"
        combined_status = agent_status
    
    return agent_response, evaluation, combined_status, eval_summary


# In[ ]:


# ===========================
# Gradio Interface
# ===========================

def create_interface():
    """Create the Gradio interface."""
    
    css = """
    .gradio-container { max-width: 1400px !important; margin: auto; }
    .response-box { background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 12px; border-radius: 8px; }
    .evaluation-box { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px; border-radius: 8px; }
    .status-box { font-family: monospace; font-size: 12px; color: #6b7280; }
    .error-box { background: #fee2e2; border-left: 4px solid #ef4444; padding: 12px; border-radius: 8px; }
    .success-indicator { color: #10b981; font-weight: bold; }
    .warning-indicator { color: #f59e0b; font-weight: bold; }
    """
    
    with gr.Blocks(
        title="AI Chatbot with Cross-Model Evaluator",
        theme=gr.themes.Soft(),
        css=css
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🤖 AI Chatbot with Cross-Model Evaluator
        ### **Agent:** Google Gemini 1.5 flash | **Evaluator:** Groq Llama 3.3 70B
        
        This system uses two different AI models:
        1. **Gemini** generates responses to your questions
        2. **Llama 70B** evaluates the quality of those responses
        """)
        
        # API Status
        if api_manager.errors:
            with gr.Group():
                gr.Markdown("### ⚠️ Setup Issues:")
                for error in api_manager.errors:
                    gr.Markdown(f"- {error}")
                gr.Markdown("""
                **To fix:**
                ```bash
                export GOOGLE_API_KEY="your-google-api-key"
                export GROQ_API_KEY="your-groq-api-key"
                ```
                Get keys from:
                - [Google AI Studio](https://makersuite.google.com/app/apikey)
                - [Groq Console](https://console.groq.com/keys)
                """)
        else:
            gr.Markdown("✅ **All API clients initialized successfully**")
        
        with gr.Row():
            # Left Column - Input Controls
            with gr.Column(scale=2):
                # Model Selection
                with gr.Group():
                    gr.Markdown("### 🎯 Model Selection")
                    agent_model = gr.Dropdown(
                        choices=list(AGENT_MODELS.keys()),
                        value="Gemini 1.5 flash",
                        label="Agent Model (Response Generator)",
                        info="Google Gemini model for generating responses"
                    )
                    
                    evaluator_model = gr.Dropdown(
                        choices=list(EVALUATOR_MODELS.keys()),
                        value="Llama 3.3 70B",
                        label="Evaluator Model",
                        info="Groq model for evaluating response quality"
                    )
                
                # User Input
                user_input = gr.Textbox(
                    lines=4,
                    placeholder="Ask me anything... For example: 'Explain quantum computing in simple terms'",
                    label="💬 Your Question",
                    max_lines=8
                )
                
                # Settings
                with gr.Group():
                    gr.Markdown("### ⚙️ Generation Settings")
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature (Creativity)",
                            info="Higher = more creative, Lower = more focused"
                        )
                        max_tokens = gr.Slider(
                            minimum=50,
                            maximum=1000,
                            value=500,
                            step=50,
                            label="Max Tokens",
                            info="Maximum response length"
                        )
                    
                    enable_evaluation = gr.Checkbox(
                        value=True,
                        label="🔍 Enable Cross-Model Evaluation",
                        info="Let Llama 70B evaluate Gemini's response"
                    )
                
                # Action Buttons
                with gr.Row():
                    generate_btn = gr.Button(
                        "🚀 Generate & Evaluate",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button("🗑️ Clear All", size="lg")
            
            # Right Column - Outputs
            with gr.Column(scale=3):
                # Quality Indicator
                quality_indicator = gr.Textbox(
                    label="📊 Response Quality",
                    interactive=False,
                    lines=1
                )
                
                # Agent Response
                with gr.Group():
                    gr.Markdown("### 🤖 Agent Response")
                    agent_output = gr.Textbox(
                        lines=10,
                        label="Gemini's Response",
                        show_copy_button=True,
                        interactive=False,
                        elem_classes=["response-box"]
                    )
                
                # Evaluation
                with gr.Group():
                    gr.Markdown("### 🔍 Evaluation Result")
                    evaluation_output = gr.Textbox(
                        lines=8,
                        label="Llama's Evaluation",
                        show_copy_button=True,
                        interactive=False,
                        elem_classes=["evaluation-box"]
                    )
                
                # Status
                status_output = gr.Textbox(
                    lines=2,
                    label="⏱️ Performance Metrics",
                    interactive=False,
                    elem_classes=["status-box"]
                )
        
        # Examples
        with gr.Row():
            gr.Examples(
                examples=[
                    ["What is the difference between machine learning and deep learning?"],
                    ["Write a Python function to calculate the factorial of a number"],
                    ["Explain the theory of relativity in simple terms"],
                    ["What are the main causes of climate change?"],
                    ["How does blockchain technology work?"],
                    ["What are the benefits and risks of artificial intelligence?"]
                ],
                inputs=user_input,
                label="💡 Example Questions"
            )
        
        # How It Works
        with gr.Accordion("ℹ️ How Cross-Model Evaluation Works", open=False):
            gr.Markdown("""
            ### The Two-Stage Process:
            
            **1. Response Generation (Gemini)**
            - Receives your question
            - Generates a comprehensive response
            - Optimized for helpfulness and accuracy
            
            **2. Quality Evaluation (Llama 70B)**
            - Analyzes the response for:
              - Accuracy and completeness
              - Clarity and coherence
              - Potential issues or biases
            - Provides feedback and improvement suggestions
            
            ### Benefits:
            - ✅ **Quality Assurance**: Second model checks for errors
            - ✅ **Bias Detection**: Different model perspectives
            - ✅ **Improvement Insights**: Specific feedback on responses
            - ✅ **Fast Processing**: API-based, no local model loading
            
            ### API Requirements:
            - Google API Key for Gemini (free tier available)
            - Groq API Key for Llama (free tier available)
            """)
        
        # Event Handlers
        generate_btn.click(
            fn=process_with_evaluation,
            inputs=[user_input, agent_model, evaluator_model, temperature, max_tokens, enable_evaluation],
            outputs=[agent_output, evaluation_output, status_output, quality_indicator]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[user_input, agent_output, evaluation_output, status_output]
        )
        
        user_input.submit(
            fn=process_with_evaluation,
            inputs=[user_input, agent_model, evaluator_model, temperature, max_tokens, enable_evaluation],
            outputs=[agent_output, evaluation_output, status_output, quality_indicator]
        )
    
    return demo


# In[ ]:


# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AI Chatbot with Cross-Model Evaluator")
    print("=" * 60)
    
    # Check API keys
    google_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not google_key:
        print("⚠️  Warning: GOOGLE_API_KEY not found")
        print("   Set it with: export GOOGLE_API_KEY='your-key-here'")
    else:
        print(f"✅ Google API Key detected: {google_key[:10]}...")
    
    if not groq_key:
        print("⚠️  Warning: GROQ_API_KEY not found")
        print("   Set it with: export GROQ_API_KEY='your-key-here'")
    else:
        print(f"✅ Groq API Key detected: {groq_key[:10]}...")
    
    print("=" * 60)
    print("📝 Starting Gradio interface...")
    print("📌 Interface will be available at: http://localhost:7860")
    print("=" * 60)
    
    # Create and launch interface
    demo = create_interface()
    demo.launch()


# In[ ]:




