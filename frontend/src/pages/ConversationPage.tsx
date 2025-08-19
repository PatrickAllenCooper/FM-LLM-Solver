import { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams, useNavigate, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { 
  ArrowLeftIcon,
  PaperAirplaneIcon,
  ChatBubbleLeftRightIcon,
  BeakerIcon,
  DocumentTextIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';
import { clsx } from 'clsx';
import { format } from 'date-fns';
import toast from 'react-hot-toast';

import { 
  StartConversationRequest
} from '@/types/api';
import { api } from '@/services/api';

// Message form schema
const MessageSchema = z.object({
  message: z.string().min(1, 'Please enter a message'),
  request_insights: z.boolean().default(false),
});

type MessageForm = z.infer<typeof MessageSchema>;

// Final generation schema
const PublishSchema = z.object({
  final_instructions: z.string().optional(),
});

type PublishForm = z.infer<typeof PublishSchema>;

export default function ConversationPage() {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showPublishDialog, setShowPublishDialog] = useState(false);

  // For new conversations
  const systemSpecId = searchParams.get('system_spec_id');
  const certificateType = searchParams.get('certificate_type') as 'lyapunov' | 'barrier' | 'inductive_invariant';
  const llmModel = searchParams.get('llm_model') || 'claude-3-5-sonnet-20241022';
  const llmMode = searchParams.get('llm_mode') || 'direct_expression';
  const temperature = parseFloat(searchParams.get('temperature') || '0.0');
  const baselineComparison = searchParams.get('baseline_comparison') === 'true';
  const isNewConversation = id === 'new' && systemSpecId && certificateType;

  // Message form
  const messageForm = useForm<MessageForm>({
    resolver: zodResolver(MessageSchema),
    defaultValues: {
      message: '',
      request_insights: false,
    },
  });

  // Publish form
  const publishForm = useForm<PublishForm>({
    resolver: zodResolver(PublishSchema),
    defaultValues: {
      final_instructions: '',
    },
  });

  // Get system spec for new conversations
  const { data: systemSpec } = useQuery({
    queryKey: ['system-spec', systemSpecId],
    queryFn: async () => {
      if (!systemSpecId) return null;
      return await api.getSystemSpecById(systemSpecId);
    },
    enabled: !!systemSpecId,
  });

  // Get existing conversation
  const { data: conversation, isLoading, error } = useQuery({
    queryKey: ['conversation', id],
    queryFn: async () => {
      if (id === 'new') return null;
      return await api.getConversation(id!);
    },
    enabled: id !== 'new',
  });

  // Start new conversation mutation
  const startConversationMutation = useMutation({
    mutationFn: async (data: StartConversationRequest) => {
      return await api.startConversation(data);
    },
    onSuccess: (newConversation) => {
      toast.success('Conversation started!');
      navigate(`/conversations/${newConversation.id}`);
    },
    onError: (error: any) => {
      toast.error(error?.message || 'Failed to start conversation');
    },
  });

  // Send message mutation
  const sendMessageMutation = useMutation({
    mutationFn: async (data: MessageForm) => {
      if (!id || id === 'new') throw new Error('No active conversation');
      return await api.sendMessage(id, {
        message: data.message,
        request_insights: data.request_insights,
      });
    },
    onSuccess: () => {
      messageForm.reset();
      queryClient.invalidateQueries({ queryKey: ['conversation', id] });
    },
    onError: (error: any) => {
      toast.error(error?.message || 'Failed to send message');
    },
  });

  // Publish certificate mutation
  const publishMutation = useMutation({
    mutationFn: async (data: PublishForm) => {
      if (!id || id === 'new') throw new Error('No active conversation');
      return await api.publishCertificateFromConversation({
        conversation_id: id,
        final_instructions: data.final_instructions,
      });
    },
    onSuccess: (result) => {
      toast.success('Certificate published successfully!');
      navigate(`/certificates/${result.candidate_id}`);
    },
    onError: (error: any) => {
      toast.error(error?.message || 'Failed to publish certificate');
    },
  });

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation?.messages]);

  // Initialize conversation for new conversations
  useEffect(() => {
    if (isNewConversation && systemSpec) {
      // Create a detailed initial message that includes specific system information
      const systemDetails = {
        name: systemSpec.name,
        type: systemSpec.system_type,
        dimension: systemSpec.dimension,
        dynamics: systemSpec.dynamics_json?.equations || 'dynamics not specified',
        domain: systemSpec.dynamics_json?.domain || 'domain not specified',
      };
      
      // Create a safe system description without potentially problematic JSON.stringify
      const dynamicsDescription = Array.isArray(systemDetails.dynamics) 
        ? systemDetails.dynamics.join(', ') 
        : typeof systemDetails.dynamics === 'string' 
        ? systemDetails.dynamics 
        : 'Complex dynamics structure';

      const domainDescription = systemDetails.domain && typeof systemDetails.domain === 'object'
        ? `Domain constraints defined`
        : 'Domain not specified';

      const initialMessage = `I'd like to explore approaches for generating a ${certificateType} function for this specific system:

**System: ${systemDetails.name}**
- Type: ${systemDetails.type} system  
- Dimension: ${systemDetails.dimension}D
- Dynamics: ${dynamicsDescription}
- ${domainDescription}

**My Configuration Preferences:**
- LLM Model: ${llmModel}
- Generation Mode: ${llmMode} 
- Temperature: ${temperature}
- Include Baseline Comparison: ${baselineComparison ? 'Yes' : 'No'}

Given these specific system properties and my research preferences, can you help me think through different mathematical strategies for ${certificateType} function construction? What approaches would work best for this particular system configuration?`;
      
      startConversationMutation.mutate({
        system_spec_id: systemSpecId!,
        certificate_type: certificateType,
        initial_message: initialMessage,
      });
    }
  }, [isNewConversation, systemSpec, systemSpecId, certificateType]);

  const onSendMessage = (data: MessageForm) => {
    sendMessageMutation.mutate(data);
  };

  const onPublishCertificate = (data: PublishForm) => {
    publishMutation.mutate(data);
    setShowPublishDialog(false);
  };

  const formatMessageTime = (timestamp: string) => {
    return format(new Date(timestamp), 'h:mm a');
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <ExclamationTriangleIcon className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-medium text-gray-900 mb-2">Conversation Not Found</h2>
          <p className="text-gray-600 mb-4">The conversation you're looking for doesn't exist or has been removed.</p>
          <Link to="/certificates" className="btn btn-primary">
            Back to Certificates
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link to="/certificates" className="btn btn-outline btn-sm">
              <ArrowLeftIcon className="w-4 h-4 mr-1" />
              Back
            </Link>
            <div>
              <h1 className="text-xl font-medium text-gray-900 flex items-center">
                <ChatBubbleLeftRightIcon className="w-5 h-5 mr-2 text-blue-600" />
                Mathematical Conversation
              </h1>
              <div className="text-sm text-gray-600">
                {systemSpec ? 
                  `${systemSpec.name} (${systemSpec.system_type}, ${systemSpec.dimension}D) • ${certificateType} function` : 
                  'Loading system specification...'
                }
                {llmModel && (
                  <div className="text-xs text-gray-500 mt-1">
                    Using {llmModel} • {llmMode} mode • Temperature: {temperature}
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {conversation?.status === 'active' && (
            <div className="flex items-center space-x-3">
              <div className="text-sm text-gray-600">
                {conversation.message_count} messages • {conversation.token_count.toLocaleString()} tokens
              </div>
              <button
                onClick={() => setShowPublishDialog(true)}
                className="btn btn-primary btn-sm"
                disabled={conversation.message_count < 3}
              >
                <BeakerIcon className="w-4 h-4 mr-1" />
                Publish Certificate
              </button>
            </div>
          )}
        </div>
        
        {conversation?.status && (
          <div className="mt-3 flex items-center space-x-4">
            <div className={clsx(
              'px-2 py-1 rounded-full text-xs font-medium',
              conversation.status === 'active' ? 'bg-green-100 text-green-800' :
              conversation.status === 'summarized' ? 'bg-blue-100 text-blue-800' :
              conversation.status === 'published' ? 'bg-purple-100 text-purple-800' :
              'bg-gray-100 text-gray-800'
            )}>
              <CheckCircleIcon className="w-3 h-3 inline mr-1" />
              {conversation.status.charAt(0).toUpperCase() + conversation.status.slice(1)}
            </div>
            
            {conversation.summary && (
              <div className="text-xs text-gray-600">
                {conversation.summary.key_insights.length} insights • 
                {conversation.summary.mathematical_approaches_discussed.length} approaches discussed
              </div>
            )}
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-4 space-y-4">
          {/* System Context Card */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="text-sm font-medium text-blue-900 mb-2 flex items-center">
              <DocumentTextIcon className="w-4 h-4 mr-1" />
              System Specification Context
            </h3>
            <div className="text-sm text-blue-800">
              {systemSpec ? (
                <>
                  <strong>{systemSpec.name}</strong> • {systemSpec.system_type} system • {systemSpec.dimension}D
                  {systemSpec.description && (
                    <div className="mt-1 text-blue-700">{systemSpec.description}</div>
                  )}
                  {systemSpec.dynamics_json?.equations && (
                    <div className="mt-2 text-xs text-blue-600">
                      <strong>Dynamics:</strong> {Array.isArray(systemSpec.dynamics_json.equations) 
                        ? systemSpec.dynamics_json.equations.join(', ') 
                        : typeof systemSpec.dynamics_json.equations === 'string'
                        ? systemSpec.dynamics_json.equations
                        : 'Complex dynamics structure'}
                    </div>
                  )}
                </>
              ) : (
                <div className="text-blue-600">Loading system specification...</div>
              )}
            </div>
          </div>

          {/* Conversation Messages */}
          {conversation?.messages?.map((message: any, index: number) => (
            <div
              key={index}
              className={clsx(
                'flex',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              <div
                className={clsx(
                  'max-w-3xl rounded-lg px-4 py-3',
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white border border-gray-200'
                )}
              >
                <div className="flex items-start justify-between mb-2">
                  <span className={clsx(
                    'text-xs font-medium',
                    message.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                  )}>
                    {message.role === 'user' ? 'You' : 'Assistant'}
                  </span>
                  <span className={clsx(
                    'text-xs ml-2',
                    message.role === 'user' ? 'text-blue-200' : 'text-gray-400'
                  )}>
                    {formatMessageTime(message.timestamp)}
                  </span>
                </div>
                <div className={clsx(
                  'text-sm whitespace-pre-wrap',
                  message.role === 'user' ? 'text-white' : 'text-gray-900'
                )}>
                  {message.content}
                </div>
                {message.metadata?.token_count && (
                  <div className={clsx(
                    'text-xs mt-2 pt-2 border-t',
                    message.role === 'user' 
                      ? 'text-blue-200 border-blue-500' 
                      : 'text-gray-400 border-gray-200'
                  )}>
                    {message.metadata.token_count} tokens
                    {message.metadata.message_type && ` • ${message.metadata.message_type}`}
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Thinking indicator */}
          {sendMessageMutation.isPending && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-lg px-4 py-3 max-w-xs">
                <div className="flex items-center space-x-2 text-gray-500">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                  <span className="text-sm">Assistant is thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Message Input */}
      {conversation?.status === 'active' && (
        <div className="bg-white border-t border-gray-200 px-6 py-4">
          <form onSubmit={messageForm.handleSubmit(onSendMessage)} className="flex items-end space-x-3">
            <div className="flex-1">
              <textarea
                {...messageForm.register('message')}
                placeholder="Ask about mathematical approaches, discuss theoretical considerations, or refine the certificate strategy..."
                className="input resize-none"
                rows={2}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    messageForm.handleSubmit(onSendMessage)();
                  }
                }}
              />
              {messageForm.formState.errors.message && (
                <p className="text-red-600 text-sm mt-1">
                  {messageForm.formState.errors.message.message}
                </p>
              )}
            </div>
            <button
              type="submit"
              disabled={sendMessageMutation.isPending || !messageForm.watch('message')}
              className="btn btn-primary"
            >
              {sendMessageMutation.isPending ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
              ) : (
                <PaperAirplaneIcon className="w-4 h-4" />
              )}
            </button>
          </form>
          
          <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
            <div>Press Enter to send, Shift+Enter for new line</div>
            <div>Mathematical notation supported</div>
          </div>
        </div>
      )}

      {/* Conversation Summary */}
      {conversation?.summary && (
        <div className="bg-orange-50 border-t border-orange-200 px-6 py-4">
          <h3 className="text-sm font-medium text-orange-900 mb-2 flex items-center">
            <SparklesIcon className="w-4 h-4 mr-1" />
            Conversation Insights
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium text-orange-800">Key Insights:</span>
              <ul className="mt-1 text-orange-700 list-disc list-inside">
                {conversation.summary.key_insights.slice(0, 3).map((insight: string, idx: number) => (
                  <li key={idx}>{insight}</li>
                ))}
              </ul>
            </div>
            <div>
              <span className="font-medium text-orange-800">Approaches Discussed:</span>
              <ul className="mt-1 text-orange-700 list-disc list-inside">
                {conversation.summary.mathematical_approaches_discussed.slice(0, 3).map((approach: string, idx: number) => (
                  <li key={idx}>{approach}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Publish Dialog */}
      {showPublishDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-lg w-full p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
              <BeakerIcon className="w-5 h-5 mr-2 text-blue-600" />
              Publish Certificate from Conversation
            </h3>
            
            <form onSubmit={publishForm.handleSubmit(onPublishCertificate)} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Final Instructions (Optional)
                </label>
                <textarea
                  {...publishForm.register('final_instructions')}
                  placeholder="Any final refinements or specific requirements for the certificate generation..."
                  className="input"
                  rows={3}
                />
                <p className="text-xs text-gray-500 mt-1">
                  These instructions will be included with the conversation context when generating the final certificate.
                </p>
              </div>

              <div className="bg-blue-50 rounded-lg p-4">
                <h4 className="text-sm font-medium text-blue-900 mb-2">What happens next:</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• LLM will synthesize the entire conversation into a formal certificate</li>
                  <li>• Generated certificate will undergo standard acceptance testing</li>
                  <li>• You'll be redirected to the certificate details page</li>
                  <li>• Conversation will be marked as published and preserved</li>
                </ul>
              </div>

              <div className="flex items-center justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => setShowPublishDialog(false)}
                  className="btn btn-outline"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={publishMutation.isPending}
                  className="btn btn-primary"
                >
                  {publishMutation.isPending ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Publishing...
                    </>
                  ) : (
                    <>
                      <BeakerIcon className="w-4 h-4 mr-2" />
                      Publish Certificate
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Status Messages */}
      {isNewConversation && !systemSpec && (
        <div className="absolute inset-0 bg-white flex items-center justify-center">
          <div className="text-center">
            <ClockIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h2 className="text-xl font-medium text-gray-900 mb-2">Loading System Specification</h2>
            <p className="text-gray-600">Preparing conversation context...</p>
          </div>
        </div>
      )}

      {conversation?.status === 'published' && (
        <div className="bg-green-50 border-t border-green-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center text-green-800">
              <CheckCircleIcon className="w-5 h-5 mr-2" />
              <span className="font-medium">Certificate Published</span>
            </div>
            {conversation.final_certificate_id && (
              <Link
                to={`/certificates/${conversation.final_certificate_id}`}
                className="btn btn-primary btn-sm"
              >
                View Certificate
              </Link>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
