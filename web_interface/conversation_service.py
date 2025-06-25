import os
import sys
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from web_interface.models import db, Conversation, ConversationMessage
from web_interface.certificate_generator import CertificateGenerator
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from inference.generate_certificate import load_finetuned_model, load_knowledge_base, retrieve_context

logger = logging.getLogger(__name__)

class ConversationService:
    """Service for managing conversational interactions with the LLM."""
    
    def __init__(self, config):
        """Initialize the conversation service."""
        self.config = config
        self.certificate_generator = CertificateGenerator(config)
        self.models = {}  # Cache for loaded models
        self.embedding_model = None
        
    def start_conversation(self, model_config: str, rag_k: int = 3) -> Dict[str, Any]:
        """Start a new conversation with the specified configuration."""
        try:
            # Create unique session ID
            session_id = str(uuid.uuid4())
            
            # Create conversation record
            conversation = Conversation(
                session_id=session_id,
                model_config=model_config,
                rag_k=rag_k,
                status='active'
            )
            
            db.session.add(conversation)
            db.session.commit()
            
            # Add initial system message
            initial_message = ConversationMessage(
                conversation_id=conversation.id,
                role='assistant',
                content="Hello! I'm here to help you generate barrier certificates for your autonomous system. "
                       "Please describe your system, including the dynamics, initial conditions, and safety requirements. "
                       "We can discuss the details and refine the description before generating the certificate.",
                message_type='chat'
            )
            
            db.session.add(initial_message)
            db.session.commit()
            
            logger.info(f"Started new conversation: {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'conversation_id': conversation.id,
                'initial_message': initial_message.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error starting conversation: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def send_message(self, session_id: str, user_message: str, message_type: str = 'chat') -> Dict[str, Any]:
        """Send a message to the conversation and get LLM response."""
        try:
            # Find conversation
            conversation = db.session.query(Conversation).filter_by(session_id=session_id).first()
            if not conversation:
                return {
                    'success': False,
                    'error': 'Conversation not found'
                }
            
            if conversation.status != 'active':
                return {
                    'success': False,
                    'error': f'Conversation is not active (status: {conversation.status})'
                }
            
            # Add user message
            user_msg = ConversationMessage(
                conversation_id=conversation.id,
                role='user',
                content=user_message.strip(),
                message_type=message_type
            )
            db.session.add(user_msg)
            
            # Generate LLM response
            start_time = datetime.utcnow()
            response_content, context_chunks = self._generate_conversation_response(
                conversation, user_message
            )
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Add assistant message
            assistant_msg = ConversationMessage(
                conversation_id=conversation.id,
                role='assistant',
                content=response_content,
                message_type='chat',
                processing_time_seconds=processing_time,
                context_chunks_used=context_chunks
            )
            db.session.add(assistant_msg)
            
            # Update conversation timestamp and extract system description if possible
            conversation.updated_at = datetime.utcnow()
            extracted_description = self._extract_system_description_from_conversation(conversation)
            if extracted_description:
                conversation.system_description = extracted_description
            
            db.session.commit()
            
            return {
                'success': True,
                'user_message': user_msg.to_dict(),
                'assistant_message': assistant_msg.to_dict(),
                'conversation_updated': True
            }
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_conversation_history(self, session_id: str) -> Dict[str, Any]:
        """Get the full conversation history."""
        try:
            conversation = db.session.query(Conversation).filter_by(session_id=session_id).first()
            if not conversation:
                return {
                    'success': False,
                    'error': 'Conversation not found'
                }
            
            messages = [msg.to_dict() for msg in conversation.messages]
            
            return {
                'success': True,
                'conversation': {
                    'id': conversation.id,
                    'session_id': conversation.session_id,
                    'status': conversation.status,
                    'model_config': conversation.model_config,
                    'rag_k': conversation.rag_k,
                    'ready_to_generate': conversation.ready_to_generate,
                    'system_description': conversation.system_description,
                    'created_at': conversation.created_at.isoformat(),
                    'updated_at': conversation.updated_at.isoformat(),
                    'message_count': conversation.message_count
                },
                'messages': messages
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def set_ready_to_generate(self, session_id: str, ready: bool) -> Dict[str, Any]:
        """Set the user's readiness to generate a certificate."""
        try:
            conversation = db.session.query(Conversation).filter_by(session_id=session_id).first()
            if not conversation:
                return {
                    'success': False,
                    'error': 'Conversation not found'
                }
            
            conversation.ready_to_generate = ready
            conversation.updated_at = datetime.utcnow()
            db.session.commit()
            
            return {
                'success': True,
                'ready_to_generate': ready
            }
            
        except Exception as e:
            logger.error(f"Error setting ready to generate: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_certificate_from_conversation(self, session_id: str) -> Dict[str, Any]:
        """Generate a barrier certificate based on the conversation context."""
        try:
            conversation = db.session.query(Conversation).filter_by(session_id=session_id).first()
            if not conversation:
                return {
                    'success': False,
                    'error': 'Conversation not found'
                }
            
            if not conversation.ready_to_generate:
                return {
                    'success': False,
                    'error': 'User has not indicated readiness to generate certificate'
                }
            
            if not conversation.system_description:
                return {
                    'success': False,
                    'error': 'No system description found in conversation'
                }
            
            # Update conversation status
            conversation.status = 'generating'
            db.session.commit()
            
            # Use the certificate generator to create the certificate
            result = self.certificate_generator.generate_certificate(
                conversation.system_description,
                conversation.model_config,
                conversation.rag_k
            )
            
            # Update conversation status
            conversation.status = 'active'  # Return to active for potential further conversation
            db.session.commit()
            
            return {
                'success': result['success'],
                'certificate_result': result
            }
            
        except Exception as e:
            logger.error(f"Error generating certificate from conversation: {str(e)}")
            # Reset conversation status
            try:
                conversation = db.session.query(Conversation).filter_by(session_id=session_id).first()
                if conversation:
                    conversation.status = 'active'
                    db.session.commit()
            except:
                pass
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_conversation_response(self, conversation: Conversation, user_message: str) -> tuple:
        """Generate a conversational response from the LLM."""
        try:
            # Load model
            model_info = self.certificate_generator._load_model(conversation.model_config)
            
            # Build conversation context
            context_messages = self._build_conversation_context(conversation, user_message)
            
            # Get RAG context if enabled
            context_chunks = 0
            rag_context = ""
            if conversation.rag_k > 0:
                try:
                    barrier_type = model_info['config']['barrier_type']
                    kb_info = self.certificate_generator._load_knowledge_base(barrier_type)
                    if kb_info:
                        embedding_model = self.certificate_generator._load_embedding_model()
                        rag_context = retrieve_context(
                            user_message,
                            embedding_model,
                            kb_info['index'],
                            kb_info['metadata'],
                            conversation.rag_k
                        )
                        context_chunks = rag_context.count('--- Context Chunk') if rag_context else 0
                except Exception as e:
                    logger.warning(f"RAG retrieval failed in conversation: {e}")
            
            # Create conversation prompt
            prompt = self._format_conversation_prompt(context_messages, rag_context)
            
            # Generate response
            result = model_info['pipeline'](prompt)
            generated_text = result[0]['generated_text']
            
            # Extract only the assistant's response
            prompt_end_marker = "[/INST]"
            output_start_index = generated_text.find(prompt_end_marker)
            if output_start_index != -1:
                response = generated_text[output_start_index + len(prompt_end_marker):].strip()
            else:
                response = generated_text.strip()
            
            # Clean up response
            response = self._clean_conversation_response(response)
            
            return response, context_chunks
            
        except Exception as e:
            logger.error(f"Error generating conversation response: {str(e)}")
            return "I apologize, but I encountered an error while processing your message. Could you please try again?", 0
    
    def _build_conversation_context(self, conversation: Conversation, current_user_message: str) -> List[Dict[str, str]]:
        """Build conversation context for the LLM prompt."""
        messages = []
        
        # Add system message
        messages.append({
            'role': 'system',
            'content': 'You are an expert in control theory and barrier certificates for autonomous systems. '
                      'You are having a conversation with a user to help them define their system and generate '
                      'appropriate barrier certificates. Be helpful, ask clarifying questions when needed, '
                      'and guide them toward a complete system description. Do not generate certificates yet - '
                      'focus on understanding and clarifying the system.'
        })
        
        # Add conversation history (recent messages)
        recent_messages = conversation.messages[-10:]  # Limit to last 10 messages for context
        for msg in recent_messages:
            if msg.role in ['user', 'assistant']:
                messages.append({
                    'role': msg.role,
                    'content': msg.content
                })
        
        # Add current user message
        messages.append({
            'role': 'user',
            'content': current_user_message
        })
        
        return messages
    
    def _format_conversation_prompt(self, messages: List[Dict[str, str]], rag_context: str = "") -> str:
        """Format conversation messages into a prompt for the LLM."""
        # Build the conversation prompt
        conversation_text = ""
        
        # Add RAG context if available
        if rag_context:
            conversation_text += f"[CONTEXT]\n{rag_context}\n[/CONTEXT]\n\n"
        
        # Add system message
        system_msg = next((msg for msg in messages if msg['role'] == 'system'), None)
        if system_msg:
            conversation_text += f"<s>[INST] {system_msg['content']}\n\n"
        else:
            conversation_text += "<s>[INST] You are an expert assistant for barrier certificate generation.\n\n"
        
        # Add conversation history
        for msg in messages:
            if msg['role'] == 'user':
                conversation_text += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                conversation_text += f"Assistant: {msg['content']}\n"
        
        conversation_text += " [/INST]"
        
        return conversation_text
    
    def _clean_conversation_response(self, response: str) -> str:
        """Clean up the LLM response for conversation."""
        # Remove common artifacts
        response = response.strip()
        
        # Remove assistant prefix if present
        if response.lower().startswith('assistant:'):
            response = response[10:].strip()
        
        # Remove any remaining conversation markers
        response = response.replace('[/INST]', '').replace('<s>', '').replace('</s>', '')
        
        return response
    
    def _extract_system_description_from_conversation(self, conversation: Conversation) -> Optional[str]:
        """Extract and construct system description from conversation messages."""
        # Look through conversation for system dynamics, initial sets, etc.
        system_parts = {
            'dynamics': None,
            'initial_set': None,
            'unsafe_set': None,
            'safe_set': None,
            'state_variables': None
        }
        
        # Simple extraction logic - look for key patterns in user messages
        for msg in conversation.messages:
            if msg.role == 'user':
                content = msg.content.lower()
                
                # Look for dynamics
                if 'dynamic' in content or 'dx/dt' in content or 'x_next' in content:
                    # Try to extract the actual dynamics
                    lines = msg.content.split('\n')
                    for line in lines:
                        if any(keyword in line.lower() for keyword in ['dynamic', 'dx/dt', 'x_next', 'x(t+1)']):
                            system_parts['dynamics'] = line.strip()
                            break
                
                # Look for initial set
                if 'initial' in content:
                    lines = msg.content.split('\n')
                    for line in lines:
                        if 'initial' in line.lower():
                            system_parts['initial_set'] = line.strip()
                            break
                
                # Look for unsafe set
                if 'unsafe' in content:
                    lines = msg.content.split('\n')
                    for line in lines:
                        if 'unsafe' in line.lower():
                            system_parts['unsafe_set'] = line.strip()
                            break
        
        # Construct description if we have at least dynamics
        if system_parts['dynamics']:
            description_parts = []
            description_parts.append(f"System Dynamics: {system_parts['dynamics']}")
            
            if system_parts['initial_set']:
                description_parts.append(f"Initial Set: {system_parts['initial_set']}")
            
            if system_parts['unsafe_set']:
                description_parts.append(f"Unsafe Set: {system_parts['unsafe_set']}")
            
            if system_parts['safe_set']:
                description_parts.append(f"Safe Set: {system_parts['safe_set']}")
            
            return '\n'.join(description_parts)
        
        return None 