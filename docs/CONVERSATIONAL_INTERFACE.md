# Conversational Interface Documentation

## Overview

The FM-LLM Solver now includes a conversational interface that allows users to have back-and-forth discussions with the LLM before generating barrier certificates. This provides a more interactive and iterative approach to system definition and certificate generation.

## Features

### Two Generation Modes

1. **Direct Generation** (Original): 
   - Provide a complete system description upfront
   - Generate certificate immediately
   - Best for users who know exactly what they want

2. **Conversational Mode** (New):
   - Start a conversation with the AI assistant
   - Discuss and refine system requirements iteratively
   - Generate certificate when ready
   - Continue conversation if certificate needs refinement

### Conversational Workflow

1. **Start Conversation**: Choose model configuration and RAG settings
2. **Chat Phase**: Discuss your system with the AI assistant
   - Describe system dynamics
   - Clarify initial conditions
   - Define safety requirements
   - Ask questions and get guidance
3. **Ready Toggle**: Check "Ready to Generate Certificate" when satisfied
4. **Generation**: Generate barrier certificate based on conversation context
5. **Decision Phase**: Accept certificate or reject and continue discussion

## Technical Implementation

### Database Models

- **Conversation**: Tracks ongoing conversations with settings and state
- **ConversationMessage**: Individual messages in the conversation
- **QueryLog**: Enhanced to link with conversations

### API Endpoints

- `POST /conversation/start` - Start new conversation
- `POST /conversation/<id>/message` - Send message
- `GET /conversation/<id>/history` - Get conversation history
- `POST /conversation/<id>/ready` - Set readiness to generate
- `POST /conversation/<id>/generate` - Generate certificate
- `POST /conversation/<id>/accept_certificate/<query_id>` - Accept certificate
- `POST /conversation/<id>/reject_certificate/<query_id>` - Reject and continue

### Frontend Components

- Mode selection radio buttons
- Chat interface with real-time messaging
- Ready to generate toggle
- Certificate acceptance/rejection controls
- Seamless integration with existing results display

## Usage Examples

### Example Conversation Flow

1. **User**: "I have a nonlinear system that I want to verify safety for"
2. **AI**: "I'd be happy to help! Can you describe the system dynamics?"
3. **User**: "dx/dt = -x^3 - y, dy/dt = x - y^3"
4. **AI**: "Great! What are the initial conditions and unsafe regions?"
5. **User**: "Initial set: x² + y² ≤ 0.1, Unsafe set: x ≥ 1.5"
6. **AI**: "Perfect! This looks like a complete system description. Are you ready to generate a barrier certificate?"
7. **User**: *Checks "Ready to Generate"* → *Clicks "Generate"*
8. Certificate generated and verified
9. **User**: Can accept or reject and continue discussion

### Benefits

- **Iterative Refinement**: Build system description step by step
- **Expert Guidance**: Get advice on system modeling
- **Error Prevention**: Catch issues before generation
- **Learning Tool**: Understand barrier certificate concepts
- **Flexibility**: Switch between modes as needed

## Configuration

The conversational interface uses the same model configurations as direct generation:
- Base models and fine-tuned variants
- RAG context retrieval (0-10 chunks)
- Verification parameter customization

## Error Handling

- Graceful degradation if conversation service fails
- Automatic fallback to error messages
- Session state management
- Connection recovery for interrupted conversations

## Future Enhancements

Potential improvements for future versions:
- Conversation persistence across sessions
- Export conversation transcripts
- Template-based system descriptions
- Multi-language support
- Voice interface integration
- Conversation analytics and insights 