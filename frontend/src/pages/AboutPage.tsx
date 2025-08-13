import { EnvelopeIcon, AcademicCapIcon, DocumentTextIcon, CpuChipIcon, BeakerIcon } from '@heroicons/react/24/outline';

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="cu-gradient-light rounded-3xl p-8 border border-primary-200">
        <div className="flex items-center space-x-4 mb-4">
          <div className="w-12 h-12 rounded-2xl cu-gradient flex items-center justify-center shadow-md">
            <CpuChipIcon className="h-6 w-6 text-cu-black" />
          </div>
          <div>
            <h1 className="academic-header text-3xl mb-0">About FM-LLM Solver</h1>
            <p className="academic-body">Rigorous evaluation of LLMs for formal verification</p>
          </div>
        </div>
        <div className="text-xs text-primary-700 font-medium">
          University of Colorado Boulder • Research Platform
        </div>
      </div>

      {/* What is FM-LLM Solver */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader">What is FM-LLM Solver?</h2>
        </div>
        <div className="card-body">
          <p className="academic-body mb-4">
            FM-LLM Solver is a comprehensive research platform designed to evaluate the capabilities of Large Language Models (LLMs) 
            in formal verification tasks. The system focuses on generating and verifying Lyapunov functions and barrier certificates 
            for dynamical systems, providing a rigorous comparison between LLM-based approaches and traditional baseline methods.
          </p>
          <p className="academic-body">
            This platform enables researchers to systematically assess how well modern AI models can contribute to the critical 
            field of formal verification, which is essential for ensuring the safety and reliability of autonomous systems.
          </p>
        </div>
      </div>

      {/* How to Use the System */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader">How to Use the System</h2>
          <p className="academic-body text-sm">Follow these steps to perform formal verification analysis</p>
        </div>
        <div className="card-body">
          <div className="space-y-6">
            
            {/* Step 1 */}
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-blue-400 to-blue-600 text-white flex items-center justify-center font-semibold shadow-md">
                  1
                </div>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center">
                  <DocumentTextIcon className="w-5 h-5 mr-2 text-blue-600" />
                  Create System Specification
                </h3>
                <p className="academic-body text-sm mb-3">
                  Define your dynamical system by providing basic information, dynamics equations, and domain constraints.
                </p>
                <div className="bg-blue-50 rounded-xl p-4 text-sm">
                  <p className="font-medium text-blue-900 mb-2">What you'll specify:</p>
                  <ul className="text-blue-800 space-y-1 text-xs">
                    <li>• System name, type (continuous/discrete/hybrid), and dimension</li>
                    <li>• State variables and their differential equations</li>
                    <li>• Domain constraints and variable bounds</li>
                    <li>• Initial and unsafe sets for verification</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-success-400 to-success-600 text-white flex items-center justify-center font-semibold shadow-md">
                  2
                </div>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center">
                  <CpuChipIcon className="w-5 h-5 mr-2 text-success-600" />
                  Generate Certificates
                </h3>
                <p className="academic-body text-sm mb-3">
                  Use the system to generate Lyapunov functions or barrier certificates through different methods.
                </p>
                <div className="bg-success-50 rounded-xl p-4 text-sm">
                  <p className="font-medium text-success-900 mb-2">Available methods:</p>
                  <ul className="text-success-800 space-y-1 text-xs">
                    <li>• <strong>LLM Direct:</strong> Direct prompting of language models</li>
                    <li>• <strong>LLM SOS:</strong> Sum-of-Squares with LLM assistance</li>
                    <li>• <strong>Baseline SOS:</strong> Traditional Sum-of-Squares methods</li>
                    <li>• <strong>Baseline SDP:</strong> Semidefinite programming approaches</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Step 3 */}
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-purple-400 to-purple-600 text-white flex items-center justify-center font-semibold shadow-md">
                  3
                </div>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center">
                  <BeakerIcon className="w-5 h-5 mr-2 text-purple-600" />
                  Analyze Results
                </h3>
                <p className="academic-body text-sm mb-3">
                  Review generated certificates, verification status, and compare different approaches.
                </p>
                <div className="bg-purple-50 rounded-xl p-4 text-sm">
                  <p className="font-medium text-purple-900 mb-2">What you can analyze:</p>
                  <ul className="text-purple-800 space-y-1 text-xs">
                    <li>• Certificate verification status and mathematical validity</li>
                    <li>• Performance comparison between LLM and baseline methods</li>
                    <li>• Execution time and computational efficiency metrics</li>
                    <li>• Success rates across different system types and complexities</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Key Features */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader">Key Features</h2>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-primary-600 rounded-full mt-2"></div>
                <div>
                  <h4 className="font-medium text-gray-900">Multi-step System Definition</h4>
                  <p className="text-sm text-gray-600">Intuitive wizard interface for defining complex dynamical systems</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-primary-600 rounded-full mt-2"></div>
                <div>
                  <h4 className="font-medium text-gray-900">LLM Integration</h4>
                  <p className="text-sm text-gray-600">Direct integration with Claude API for certificate generation</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-primary-600 rounded-full mt-2"></div>
                <div>
                  <h4 className="font-medium text-gray-900">Comprehensive Validation</h4>
                  <p className="text-sm text-gray-600">Intelligent form validation with detailed error feedback</p>
                </div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-primary-600 rounded-full mt-2"></div>
                <div>
                  <h4 className="font-medium text-gray-900">Baseline Comparison</h4>
                  <p className="text-sm text-gray-600">Compare LLM performance against traditional methods</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-primary-600 rounded-full mt-2"></div>
                <div>
                  <h4 className="font-medium text-gray-900">Real-time Verification</h4>
                  <p className="text-sm text-gray-600">Automatic verification of generated certificates</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-primary-600 rounded-full mt-2"></div>
                <div>
                  <h4 className="font-medium text-gray-900">Research Analytics</h4>
                  <p className="text-sm text-gray-600">Detailed metrics and performance analysis tools</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Author Information */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader">Author Information</h2>
        </div>
        <div className="card-body">
          <div className="flex items-start space-x-6">
            <div className="flex-shrink-0">
              <div className="w-16 h-16 rounded-2xl cu-gradient flex items-center justify-center shadow-lg">
                <AcademicCapIcon className="h-8 w-8 text-cu-black" />
              </div>
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Patrick Cooper</h3>
              <p className="academic-body mb-4">
                Research developer and creator of the FM-LLM Solver platform. This system was developed as part of 
                ongoing research into the application of Large Language Models for formal verification tasks at the 
                University of Colorado Boulder.
              </p>
              <div className="flex items-center space-x-4">
                <a 
                  href="mailto:patrick.cooper@colorado.edu"
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-primary-700 bg-primary-50 border border-primary-200 rounded-xl hover:bg-primary-100 transition-colors duration-200"
                >
                  <EnvelopeIcon className="w-4 h-4 mr-2" />
                  patrick.cooper@colorado.edu
                </a>
                <div className="text-sm text-gray-500">
                  University of Colorado Boulder
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Technical Information */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader">Technical Information</h2>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Frontend Technologies</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• React with TypeScript</li>
                <li>• Material Design 3 with Tailwind CSS</li>
                <li>• React Hook Form with Zod validation</li>
                <li>• TanStack Query for state management</li>
                <li>• Vite for build tooling</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Backend Technologies</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Node.js with Express.js</li>
                <li>• PostgreSQL with Knex.js</li>
                <li>• Anthropic Claude API integration</li>
                <li>• JWT authentication</li>
                <li>• Docker containerization</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Getting Started */}
      <div className="cu-gradient-light border border-primary-200 rounded-2xl p-6">
        <div className="text-center">
          <h3 className="text-lg font-semibold text-primary-900 mb-2">Ready to Get Started?</h3>
          <p className="text-primary-800 text-sm mb-4">
            Begin by creating your first system specification and exploring the formal verification capabilities.
          </p>
          <a 
            href="/system-specs/create"
            className="btn-primary inline-flex items-center"
          >
            <DocumentTextIcon className="w-4 h-4 mr-2" />
            Create System Specification
          </a>
        </div>
      </div>
    </div>
  );
}
