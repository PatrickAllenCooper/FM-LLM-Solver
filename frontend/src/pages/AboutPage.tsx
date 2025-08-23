import { 
  EnvelopeIcon, 
  AcademicCapIcon, 
  DocumentTextIcon, 
  CpuChipIcon, 
  BeakerIcon,
  CalculatorIcon,
  ChartBarIcon,
  Cog8ToothIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/outline';

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
            <p className="academic-body">Rigorous evaluation of LLMs for formal methods and stability analysis</p>
          </div>
        </div>
        <div className="text-xs text-primary-700 font-medium">
          University of Colorado Boulder • Research Platform • Version 2.0.1
        </div>
        <div className="mt-3 text-sm text-primary-600">
          <strong>Latest Enhancements (v2.0.1):</strong> Security hardening with zero vulnerabilities, 
          ESLint integration for code quality, production-ready logging, and comprehensive technical analysis 
          with experimental parameter controls for research transparency.
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
            in formal methods and stability analysis tasks. The system focuses on generating and validating Lyapunov function candidates 
            and barrier certificate candidates for dynamical systems, providing a rigorous comparison between LLM-based approaches and 
            traditional baseline methods through a two-stage acceptance protocol.
          </p>
          <p className="academic-body">
            This platform enables researchers to systematically assess how well modern AI models can contribute to the critical 
            field of formal methods and stability analysis, which is essential for ensuring the safety and reliability of autonomous systems.
          </p>
        </div>
      </div>

      {/* Mathematical Foundations */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader flex items-center">
            <CalculatorIcon className="w-6 h-6 mr-3 text-primary-600" />
            Mathematical Foundations
          </h2>
          <p className="academic-body text-sm">Core mathematical concepts and formal methods</p>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            {/* Lyapunov Theory */}
            <div className="space-y-4">
              <div className="bg-blue-50 rounded-xl p-6">
                <h4 className="text-lg font-semibold text-blue-900 mb-3 flex items-center">
                  <div className="w-8 h-8 rounded-lg bg-blue-200 flex items-center justify-center mr-3">
                    <span className="text-blue-800 font-bold text-sm">V</span>
                  </div>
                  Lyapunov Function Theory
                </h4>
                <p className="text-blue-800 text-sm mb-4">
                  Stability analysis through energy-like functions that prove system convergence to equilibrium states.
                </p>
                <div className="space-y-3 text-xs">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-blue-900 mb-1">Conditions:</p>
                    <ul className="text-blue-800 space-y-1">
                      <li>• <strong>Positive definite:</strong> V(x) &gt; 0 for x &ne; 0</li>
                      <li>• <strong>Zero at equilibrium:</strong> V(0) = 0</li>
                      <li>• <strong>Decreasing:</strong> V&#775;(x) &le; 0 along trajectories</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-blue-900 mb-1">Applications:</p>
                    <p className="text-blue-800">Autonomous systems, control theory, robotics stability</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Barrier Certificate Theory */}
            <div className="space-y-4">
              <div className="bg-green-50 rounded-xl p-6">
                <h4 className="text-lg font-semibold text-green-900 mb-3 flex items-center">
                  <div className="w-8 h-8 rounded-lg bg-green-200 flex items-center justify-center mr-3">
                    <span className="text-green-800 font-bold text-sm">B</span>
                  </div>
                  Barrier Certificate Theory
                </h4>
                <p className="text-green-800 text-sm mb-4">
                  Safety verification through functions that maintain separation between safe and unsafe regions.
                </p>
                <div className="space-y-3 text-xs">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-green-900 mb-1">Conditions:</p>
                    <ul className="text-green-800 space-y-1">
                      <li>• <strong>Initial safety:</strong> B(x₀) &le; -m_init</li>
                      <li>• <strong>Unsafe separation:</strong> B(x_unsafe) &ge; m_unsafe</li>
                      <li>• <strong>Invariant:</strong> B&#775;(x) &le; 0 on boundary</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-green-900 mb-1">Applications:</p>
                    <p className="text-green-800">Safety-critical systems, collision avoidance, reachability</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Formal Methods */}
          <div className="mt-8 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-xl p-6 border border-purple-200">
            <h4 className="text-lg font-semibold text-purple-900 mb-4 flex items-center">
              <ShieldCheckIcon className="w-6 h-6 mr-3 text-purple-600" />
              Formal Methods Integration
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white rounded-lg p-4">
                <h5 className="font-medium text-purple-900 mb-2">Sum-of-Squares (SOS)</h5>
                <p className="text-purple-800 text-sm mb-2">
                  Polynomial optimization for rigorous certificate synthesis using semidefinite programming.
                </p>
                <div className="text-xs text-purple-700">
                  <strong>Method:</strong> Express polynomials as sums of squares to guarantee non-negativity
                </div>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h5 className="font-medium text-purple-900 mb-2">SMT/δ-SAT Solving</h5>
                <p className="text-purple-800 text-sm mb-2">
                  Satisfiability modulo theories for non-polynomial constraints with bounded precision.
                </p>
                <div className="text-xs text-purple-700">
                  <strong>Tools:</strong> dReal, Z3 for hybrid system verification
                </div>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h5 className="font-medium text-purple-900 mb-2">Reachability Analysis</h5>
                <p className="text-purple-800 text-sm mb-2">
                  Forward/backward reachable set computation for safety verification cross-checks.
                </p>
                <div className="text-xs text-purple-700">
                  <strong>Tools:</strong> Flow*, CORA for continuous systems
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Acceptance Protocol */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader flex items-center">
            <Cog8ToothIcon className="w-6 h-6 mr-3 text-primary-600" />
            Two-Stage Acceptance Protocol
          </h2>
          <p className="academic-body text-sm">Rigorous numerical and formal validation methodology</p>
        </div>
        <div className="card-body">
          <div className="space-y-8">
            
            {/* Stage A - Numerical */}
            <div className="border-l-4 border-orange-400 pl-6">
              <h4 className="text-lg font-semibold text-gray-900 mb-4">Stage A: Numerical Validation (Always Applied)</h4>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="bg-orange-50 rounded-xl p-5">
                    <h5 className="font-medium text-orange-900 mb-3">Sampling Strategies</h5>
                    <ul className="text-orange-800 text-sm space-y-2">
                      <li>• <strong>Sobol/LHS sequences:</strong> Low-discrepancy sampling across domain</li>
                      <li>• <strong>Boundary oversampling:</strong> Dense sampling near B(x) = 0</li>
                      <li>• <strong>Equilibrium focus:</strong> Concentrated sampling around critical points</li>
                      <li>• <strong>Adaptive refinement:</strong> Iterative refinement in violation regions</li>
                    </ul>
                  </div>
                  
                  <div className="bg-orange-50 rounded-xl p-5">
                    <h5 className="font-medium text-orange-900 mb-3">Gradient Analysis</h5>
                    <ul className="text-orange-800 text-sm space-y-2">
                      <li>• <strong>Automatic differentiation:</strong> &nabla;V, &nabla;B computation</li>
                      <li>• <strong>Lie derivatives:</strong> V&#775; = &nabla;V&middot;f for continuous systems</li>
                      <li>• <strong>Discrete differences:</strong> &Delta; = V&compfn;f - V for discrete systems</li>
                    </ul>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="bg-orange-50 rounded-xl p-5">
                    <h5 className="font-medium text-orange-900 mb-3">Margin Requirements</h5>
                    <div className="space-y-3 text-sm">
                      <div className="bg-white rounded-lg p-3">
                        <p className="font-medium text-orange-900">Lyapunov Margins:</p>
                        <ul className="text-orange-800 mt-1 space-y-1">
                          <li>• Positivity: V(x) &ge; &epsilon;_pos&Vert;x&Vert;^p</li>
                          <li>• Decrease: V&#775;(x) &le; -&epsilon;_dec&Vert;x&Vert;^p</li>
                        </ul>
                      </div>
                      <div className="bg-white rounded-lg p-3">
                        <p className="font-medium text-orange-900">Barrier Margins:</p>
                        <ul className="text-orange-800 mt-1 space-y-1">
                          <li>• Initial: B &le; -m_init on I₀</li>
                          <li>• Unsafe: B &ge; m_unsafe on X_u</li>
                          <li>• Invariant: L_fB &le; -m_inv on &part;B</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Stage B - Formal */}
            <div className="border-l-4 border-blue-400 pl-6">
              <h4 className="text-lg font-semibold text-gray-900 mb-4">Stage B: Formal Verification (Selective Application)</h4>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="bg-blue-50 rounded-xl p-4">
                  <h5 className="font-medium text-blue-900 mb-2">Polynomial Systems</h5>
                  <p className="text-blue-800 text-sm mb-3">
                    Sum-of-Squares decomposition with Lyapunov/barrier multipliers via semidefinite programming.
                  </p>
                  <div className="bg-blue-100 rounded-lg p-3 text-xs">
                    <strong>Solvers:</strong> SeDuMi, MOSEK, CVXOPT
                  </div>
                </div>
                
                <div className="bg-blue-50 rounded-xl p-4">
                  <h5 className="font-medium text-blue-900 mb-2">Non-polynomial Systems</h5>
                  <p className="text-blue-800 text-sm mb-3">
                    SMT solving with δ-satisfiability for transcendental and hybrid constraints.
                  </p>
                  <div className="bg-blue-100 rounded-lg p-3 text-xs">
                    <strong>Tools:</strong> dReal (δ-complete), Z3 (approximation)
                  </div>
                </div>
                
                <div className="bg-blue-50 rounded-xl p-4">
                  <h5 className="font-medium text-blue-900 mb-2">Cross-validation</h5>
                  <p className="text-blue-800 text-sm mb-3">
                    Reachability analysis cross-check on representative system subsets.
                  </p>
                  <div className="bg-blue-100 rounded-lg p-3 text-xs">
                    <strong>Tools:</strong> Flow*, CORA, SpaceEx
                  </div>
                </div>
              </div>

              <div className="mt-6 bg-blue-100 rounded-xl p-4">
                <p className="text-blue-900 font-medium text-sm mb-2">Acceptance Criteria:</p>
                <p className="text-blue-800 text-sm">
                  A candidate is <strong>accepted</strong> if and only if it passes Stage A numerical validation 
                  <strong>and</strong> passes Stage B formal verification when Stage B is enabled for the given system class.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Research Methodology */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader flex items-center">
            <ChartBarIcon className="w-6 h-6 mr-3 text-primary-600" />
            Research Methodology & Statistical Analysis
          </h2>
          <p className="academic-body text-sm">Rigorous experimental design and statistical validation</p>
        </div>
        <div className="card-body">
          <div className="space-y-6">
            
            {/* Experimental Design */}
            <div className="bg-gray-50 rounded-xl p-6">
              <h4 className="text-lg font-semibold text-gray-900 mb-4">Pre-registerable Experimental Design</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h5 className="font-medium text-gray-900 mb-3">Primary Research Questions</h5>
                  <ul className="text-gray-700 text-sm space-y-2">
                    <li>• <strong>Success Rate Analysis:</strong> LLM vs classical baselines with equal computational budgets</li>
                    <li>• <strong>Temporal Efficiency:</strong> Time-to-acceptance comparison across methods</li>
                    <li>• <strong>Solution Quality:</strong> Margins, accepted region size, candidate complexity</li>
                    <li>• <strong>Robustness Assessment:</strong> Performance under prompt variants and parameter perturbations</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-gray-900 mb-3">Experimental Conditions</h5>
                  <ul className="text-gray-700 text-sm space-y-2">
                    <li>• <strong>LLM Modes:</strong> Direct expression, basis+coefficients, structure+constraints, conversational dialogue</li>
                    <li>• <strong>Budget Controls:</strong> Max LLM calls, tokens, solver CPU time, restarts</li>
                    <li>• <strong>Data Splits:</strong> Development set for prompt tuning, test set frozen before analysis</li>
                    <li>• <strong>Baseline Methods:</strong> SOS, SDP, quadratic templates with matching budgets</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Advanced Research Features */}
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
              <h4 className="text-lg font-semibold text-blue-900 mb-4 flex items-center">
                <BeakerIcon className="w-5 h-5 mr-2" />
                Advanced Research Capabilities
              </h4>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h5 className="font-medium text-blue-900 mb-3">Conversational Mode</h5>
                  <p className="text-blue-800 text-sm mb-3">
                    Engage in iterative mathematical dialogue with Claude before generating certificates. 
                    Explore theoretical approaches, discuss system properties, and refine understanding through conversation.
                  </p>
                  <div className="bg-blue-100 rounded-lg p-3 text-xs">
                    <strong>Features:</strong> Multi-turn dialogue, automatic summarization, context-aware generation, mathematical reasoning exploration
                  </div>
                </div>
                
                <div>
                  <h5 className="font-medium text-blue-900 mb-3">Technical Analysis & Parameter Controls</h5>
                  <p className="text-blue-800 text-sm mb-3">
                    Complete transparency into numerical validation with detailed technical metrics, 
                    violation analysis, and runtime parameter adjustment for experimental research.
                  </p>
                  <div className="bg-blue-100 rounded-lg p-3 text-xs">
                    <strong>Controls:</strong> Sample count (100-10K), sampling methods, tolerance settings, re-run capabilities for sensitivity analysis
                  </div>
                </div>
              </div>

              <div className="mt-4 p-4 bg-white rounded-lg border border-blue-200">
                <p className="text-blue-800 text-sm">
                  <strong>Research Applications:</strong> Study LLM mathematical reasoning, parameter sensitivity analysis, 
                  approach comparison, educational insights, and iterative certificate refinement for publication-quality research.
                </p>
              </div>
            </div>

            {/* Statistical Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-green-50 rounded-xl p-5">
                <h5 className="font-medium text-green-900 mb-3">Success Rate Analysis</h5>
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-green-900 text-sm mb-1">McNemar's Test (Paired)</p>
                    <p className="text-green-800 text-xs">
                      Non-parametric test for comparing paired binary outcomes between LLM and baseline methods 
                      on identical system specifications.
                    </p>
                  </div>
                  <div className="text-green-800 text-xs">
                    <strong>Null Hypothesis:</strong> P(LLM success | baseline fail) = P(baseline success | LLM fail)
                  </div>
                </div>
              </div>

              <div className="bg-purple-50 rounded-xl p-5">
                <h5 className="font-medium text-purple-900 mb-3">Performance Metrics</h5>
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-purple-900 text-sm mb-1">Wilcoxon Signed-Rank Test</p>
                    <p className="text-purple-800 text-xs">
                      Non-parametric paired test for continuous metrics (time, quality) with 
                      median [IQR] reporting and 95% CI via bootstrap.
                    </p>
                  </div>
                  <div className="text-purple-800 text-xs">
                    <strong>Metrics:</strong> Execution time, margin strength, complexity measures
                  </div>
                </div>
              </div>
            </div>

            {/* Quality Metrics */}
            <div className="bg-indigo-50 rounded-xl p-6 border border-indigo-200">
              <h5 className="font-medium text-indigo-900 mb-4">Comprehensive Quality Assessment</h5>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4">
                  <h6 className="font-medium text-indigo-900 mb-2">Strength Metrics</h6>
                  <ul className="text-indigo-800 text-xs space-y-1">
                    <li>• <strong>Lyapunov:</strong> min_x V&#775;/&Vert;x&Vert;^p</li>
                    <li>• <strong>Barrier:</strong> m_init, m_unsafe, m_inv</li>
                    <li>• <strong>Region size:</strong> Largest accepted sublevel &Omega;_c</li>
                    <li>• <strong>Safety volume:</strong> Monte-Carlo estimates</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <h6 className="font-medium text-indigo-900 mb-2">Complexity Metrics</h6>
                  <ul className="text-indigo-800 text-xs space-y-1">
                    <li>• <strong>Polynomial degree</strong></li>
                    <li>• <strong>Term count</strong></li>
                    <li>• <strong>AST size and depth</strong></li>
                    <li>• <strong>Coefficient magnitudes</strong></li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <h6 className="font-medium text-indigo-900 mb-2">Robustness Tests</h6>
                  <ul className="text-indigo-800 text-xs space-y-1">
                    <li>• <strong>Parameter jitter:</strong> Pass rate under perturbation</li>
                    <li>• <strong>Gradient bounds:</strong> Violation sensitivity analysis</li>
                    <li>• <strong>Prompt variants:</strong> Consistency across reformulations</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Reproducibility & Provenance */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader flex items-center">
            <DocumentTextIcon className="w-6 h-6 mr-3 text-primary-600" />
            Reproducibility & Research Provenance
          </h2>
          <p className="academic-body text-sm">Complete experimental traceability and reproducible research infrastructure</p>
        </div>
        <div className="card-body">
          <div className="space-y-6">
            
            {/* Provenance Tracking */}
            <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
              <h4 className="text-lg font-semibold text-blue-900 mb-4">Complete Experimental Provenance</h4>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-blue-900 mb-2">System Specification Tracking</h5>
                    <ul className="text-blue-800 text-sm space-y-1">
                      <li>• <strong>Immutable hashing:</strong> SHA-256 of SystemSpec JSON for version control</li>
                      <li>• <strong>Schema versioning:</strong> Spec version tracking with backward compatibility</li>
                      <li>• <strong>Created by tracking:</strong> User provenance for all generated specifications</li>
                      <li>• <strong>Modification history:</strong> Complete audit trail of system changes</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-blue-900 mb-2">LLM Interaction Logging</h5>
                    <ul className="text-blue-800 text-sm space-y-1">
                      <li>• <strong>Model versioning:</strong> Anthropic Claude model and version tracking</li>
                      <li>• <strong>Prompt preservation:</strong> Complete prompt text with system context</li>
                      <li>• <strong>Response archival:</strong> Full LLM outputs with token counts</li>
                      <li>• <strong>Temperature settings:</strong> Reproducible generation parameters</li>
                    </ul>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-blue-900 mb-2">Acceptance Analysis Metadata</h5>
                    <ul className="text-blue-800 text-sm space-y-1">
                      <li>• <strong>Sampling parameters:</strong> Method, count, tolerance, random seeds</li>
                      <li>• <strong>Tool versions:</strong> Numerical libraries, solver versions, platform details</li>
                      <li>• <strong>Execution environment:</strong> Hardware specs, compute resources used</li>
                      <li>• <strong>Timing data:</strong> Stage A/B execution duration with breakdown</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-blue-900 mb-2">Artifact Persistence</h5>
                    <ul className="text-blue-800 text-sm space-y-1">
                      <li>• <strong>Immutable storage:</strong> Google Cloud Storage for formal verification artifacts</li>
                      <li>• <strong>SOS certificates:</strong> SDP problem matrices and solutions</li>
                      <li>• <strong>SMT logs:</strong> Complete solver traces and δ-satisfiability proofs</li>
                      <li>• <strong>Reachability results:</strong> Flow*/CORA analysis outputs and visualizations</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Code & Infrastructure Versioning */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-50 rounded-xl p-5">
                <h5 className="font-medium text-gray-900 mb-3">Research Infrastructure Integrity</h5>
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-gray-900 text-sm mb-1">Code Commit Tracking</p>
                    <p className="text-gray-800 text-xs">
                      Every experimental run records the exact Git commit SHA of the codebase used, 
                      ensuring reproducibility of computational results across software versions.
                    </p>
                  </div>
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-gray-900 text-sm mb-1">Dependency Management</p>
                    <p className="text-gray-800 text-xs">
                      Complete package.json and requirements.txt lockfiles preserved with exact 
                      versions of all mathematical libraries and computational dependencies.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 rounded-xl p-5">
                <h5 className="font-medium text-yellow-900 mb-3">Experimental Re-execution</h5>
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-yellow-900 text-sm mb-1">Parameter Controls</p>
                    <p className="text-yellow-800 text-xs">
                      Runtime parameter adjustment enables sensitivity analysis: sample counts, 
                      tolerances, sampling methods can be modified and re-run with full provenance.
                    </p>
                  </div>
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-yellow-900 text-sm mb-1">Deterministic Seeds</p>
                    <p className="text-yellow-800 text-xs">
                      All random number generation uses fixed seeds, ensuring identical sampling 
                      patterns and numerical results across experimental replications.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Data Integrity & Consistency */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader flex items-center">
            <ShieldCheckIcon className="w-6 h-6 mr-3 text-primary-600" />
            Data Integrity & Consistency Requirements
          </h2>
          <p className="academic-body text-sm">Rigorous data validation and consistency enforcement for research reliability</p>
        </div>
        <div className="card-body">
          <div className="space-y-6">
            
            {/* Mathematical Correctness */}
            <div className="bg-red-50 rounded-xl p-6 border border-red-200">
              <h4 className="text-lg font-semibold text-red-900 mb-4">Mathematical Expression Evaluation Correctness</h4>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-red-900 mb-2">Core Requirements</h5>
                    <ul className="text-red-800 text-sm space-y-1">
                      <li>• <strong>Negative number handling:</strong> x₁² + x₂² with x₁=-1.5, x₂=-2.4 MUST evaluate to 8.01</li>
                      <li>• <strong>Never zero for positive expressions:</strong> Expressions containing only squares and positive terms cannot evaluate to 0</li>
                      <li>• <strong>Domain compliance:</strong> All sampling within specified bounds (e.g., [-5,5] for test systems)</li>
                      <li>• <strong>Precision maintenance:</strong> Numerical accuracy to specified tolerance levels</li>
                    </ul>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-red-900 mb-2">Validation Pipeline</h5>
                    <ul className="text-red-800 text-sm space-y-1">
                      <li>• <strong>Tokenizer verification:</strong> Proper parsing of negative numbers in expressions</li>
                      <li>• <strong>Evaluation testing:</strong> Automated test cases for mathematical correctness</li>
                      <li>• <strong>Edge case handling:</strong> Boundary conditions and limit cases</li>
                      <li>• <strong>Cross-validation:</strong> Multiple evaluation methods for consistency</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Status-Violation Consistency */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-green-50 rounded-xl p-5 border border-green-200">
                <h5 className="font-medium text-green-900 mb-3">Acceptance Status Consistency</h5>
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-green-900 text-sm mb-1">Strict Invariants</p>
                    <ul className="text-green-800 text-xs space-y-1">
                      <li>• <strong>Accepted certificates:</strong> MUST have violations = 0</li>
                      <li>• <strong>Failed certificates:</strong> MUST have violations > 0</li>
                      <li>• <strong>No contradictions:</strong> Cannot be accepted with violations present</li>
                      <li>• <strong>Status updates:</strong> Real-time consistency checks on re-evaluation</li>
                    </ul>
                  </div>
                  <div className="bg-green-100 rounded-lg p-3 text-xs">
                    <strong>Database Constraints:</strong> Enforced at application and database levels
                  </div>
                </div>
              </div>

              <div className="bg-purple-50 rounded-xl p-5 border border-purple-200">
                <h5 className="font-medium text-purple-900 mb-3">Technical Details Accessibility</h5>
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-purple-900 text-sm mb-1">Universal Visibility</p>
                    <ul className="text-purple-800 text-xs space-y-1">
                      <li>• <strong>All certificates:</strong> Technical details button always visible</li>
                      <li>• <strong>Debug access:</strong> Failed certificates show complete violation analysis</li>
                      <li>• <strong>Parameter controls:</strong> Available regardless of acceptance status</li>
                      <li>• <strong>Research transparency:</strong> Full technical analysis for all outcomes</li>
                    </ul>
                  </div>
                  <div className="bg-purple-100 rounded-lg p-3 text-xs">
                    <strong>Research Requirement:</strong> No hidden information based on status
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Computational Methodology & Scalability */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader flex items-center">
            <CpuChipIcon className="w-6 h-6 mr-3 text-primary-600" />
            Computational Methodology & Scalability
          </h2>
          <p className="academic-body text-sm">Advanced numerical methods and high-performance computational strategies</p>
        </div>
        <div className="card-body">
          <div className="space-y-6">
            
            {/* Numerical Methods */}
            <div className="bg-indigo-50 rounded-xl p-6 border border-indigo-200">
              <h4 className="text-lg font-semibold text-indigo-900 mb-4">Advanced Numerical Validation Methods</h4>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-medium text-indigo-900 mb-2">Quasi-Monte Carlo Sampling</h5>
                  <ul className="text-indigo-800 text-sm space-y-1">
                    <li>• <strong>Sobol sequences:</strong> Low-discrepancy sampling for uniform domain coverage</li>
                    <li>• <strong>Latin Hypercube:</strong> Stratified sampling for parameter space exploration</li>
                    <li>• <strong>Adaptive refinement:</strong> Iterative density increases in violation regions</li>
                    <li>• <strong>Boundary oversampling:</strong> Focused sampling near decision boundaries</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-medium text-indigo-900 mb-2">Gradient-Based Analysis</h5>
                  <ul className="text-indigo-800 text-sm space-y-1">
                    <li>• <strong>Automatic differentiation:</strong> Forward/reverse mode for exact gradients</li>
                    <li>• <strong>Lie derivative computation:</strong> V̇ = ∇V·f(x) for continuous systems</li>
                    <li>• <strong>Discrete differences:</strong> Δ = V∘f(x) - V(x) for discrete systems</li>
                    <li>• <strong>Sensitivity analysis:</strong> Parameter perturbation bounds</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-medium text-indigo-900 mb-2">Adversarial Optimization</h5>
                  <ul className="text-indigo-800 text-sm space-y-1">
                    <li>• <strong>Violation maximization:</strong> Gradient ascent to find worst-case points</li>
                    <li>• <strong>Counterexample search:</strong> Active violation discovery algorithms</li>
                    <li>• <strong>Basin exploration:</strong> Multiple initialization for global optimization</li>
                    <li>• <strong>Constraint handling:</strong> Projected gradients for domain boundaries</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Scalability & Performance */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-50 rounded-xl p-5">
                <h5 className="font-medium text-gray-900 mb-3">Computational Scalability</h5>
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-gray-900 text-sm mb-1">Parallel Processing</p>
                    <ul className="text-gray-800 text-xs space-y-1">
                      <li>• <strong>Batch evaluation:</strong> Vectorized mathematical operations</li>
                      <li>• <strong>Worker processes:</strong> Multi-core numerical computation</li>
                      <li>• <strong>Asynchronous validation:</strong> Non-blocking acceptance checking</li>
                      <li>• <strong>Cloud scalability:</strong> Horizontal scaling on Cloud Run</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-gray-900 text-sm mb-1">Memory Management</p>
                    <ul className="text-gray-800 text-xs space-y-1">
                      <li>• <strong>Streaming evaluation:</strong> Large sample sets without memory overflow</li>
                      <li>• <strong>Garbage collection:</strong> Optimized for mathematical computation</li>
                      <li>• <strong>Cache optimization:</strong> Expression compilation and reuse</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-orange-50 rounded-xl p-5">
                <h5 className="font-medium text-orange-900 mb-3">Performance Optimization</h5>
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-orange-900 text-sm mb-1">Expression Compilation</p>
                    <ul className="text-orange-800 text-xs space-y-1">
                      <li>• <strong>Just-in-time compilation:</strong> Mathematical expression optimization</li>
                      <li>• <strong>Symbolic preprocessing:</strong> Expression simplification and canonicalization</li>
                      <li>• <strong>Common subexpression elimination:</strong> Shared computation reuse</li>
                      <li>• <strong>Numerical stability:</strong> Condition number monitoring</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-orange-900 text-sm mb-1">Algorithm Selection</p>
                    <ul className="text-orange-800 text-xs space-y-1">
                      <li>• <strong>Adaptive methods:</strong> Algorithm choice based on system properties</li>
                      <li>• <strong>Convergence monitoring:</strong> Early termination for efficiency</li>
                      <li>• <strong>Precision vs speed:</strong> Configurable accuracy-performance trade-offs</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Security & Authentication */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader flex items-center">
            <ShieldCheckIcon className="w-6 h-6 mr-3 text-primary-600" />
            Security & Authentication Architecture
          </h2>
          <p className="academic-body text-sm">Research-grade security with comprehensive access controls and audit trails</p>
        </div>
        <div className="card-body">
          <div className="space-y-6">
            
            {/* Authentication & Authorization */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-blue-50 rounded-xl p-5 border border-blue-200">
                <h5 className="font-medium text-blue-900 mb-3">Multi-Layer Authentication</h5>
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-blue-900 text-sm mb-1">Firebase Authentication</p>
                    <ul className="text-blue-800 text-xs space-y-1">
                      <li>• <strong>Email/password:</strong> Secure credential management</li>
                      <li>• <strong>JWT tokens:</strong> Stateless session management</li>
                      <li>• <strong>Token refresh:</strong> Automatic re-authentication</li>
                      <li>• <strong>Session security:</strong> HTTPS-only, secure cookies</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-blue-900 text-sm mb-1">Role-Based Access Control</p>
                    <ul className="text-blue-800 text-xs space-y-1">
                      <li>• <strong>Admin:</strong> User provisioning, system configuration</li>
                      <li>• <strong>Researcher:</strong> Full system and certificate creation</li>
                      <li>• <strong>Viewer:</strong> Read-only access to results</li>
                      <li>• <strong>Email authorization:</strong> Admin-controlled user approval</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 rounded-xl p-5 border border-green-200">
                <h5 className="font-medium text-green-900 mb-3">API Security & Secret Management</h5>
                <div className="space-y-3">
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-green-900 text-sm mb-1">Google Secret Manager</p>
                    <ul className="text-green-800 text-xs space-y-1">
                      <li>• <strong>API key storage:</strong> Anthropic Claude keys in encrypted storage</li>
                      <li>• <strong>Database credentials:</strong> Secure connection string management</li>
                      <li>• <strong>Rotation support:</strong> Automated key rotation capabilities</li>
                      <li>• <strong>Access logging:</strong> Complete audit trail for secret access</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-3">
                    <p className="font-medium text-green-900 text-sm mb-1">Network Security</p>
                    <ul className="text-green-800 text-xs space-y-1">
                      <li>• <strong>HTTPS enforcement:</strong> TLS 1.3 with perfect forward secrecy</li>
                      <li>• <strong>CORS policies:</strong> Restricted cross-origin access</li>
                      <li>• <strong>Rate limiting:</strong> DDoS protection and abuse prevention</li>
                      <li>• <strong>Request validation:</strong> Input sanitization and schema validation</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Audit & Compliance */}
            <div className="bg-purple-50 rounded-xl p-6 border border-purple-200">
              <h4 className="text-lg font-semibold text-purple-900 mb-4">Comprehensive Audit Trail</h4>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-medium text-purple-900 mb-2">User Activity Logging</h5>
                  <ul className="text-purple-800 text-sm space-y-1">
                    <li>• <strong>Authentication events:</strong> Login/logout with IP tracking</li>
                    <li>• <strong>System creation:</strong> All specification changes logged</li>
                    <li>• <strong>Certificate generation:</strong> LLM interactions and parameters</li>
                    <li>• <strong>Admin actions:</strong> User provisioning and role changes</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-medium text-purple-900 mb-2">Research Data Integrity</h5>
                  <ul className="text-purple-800 text-sm space-y-1">
                    <li>• <strong>Immutable records:</strong> Append-only experimental data</li>
                    <li>• <strong>Cryptographic hashing:</strong> Tamper-evident data storage</li>
                    <li>• <strong>Backup verification:</strong> Regular integrity checks</li>
                    <li>• <strong>Chain of custody:</strong> Complete data lineage tracking</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-medium text-purple-900 mb-2">Compliance & Privacy</h5>
                  <ul className="text-purple-800 text-sm space-y-1">
                    <li>• <strong>GDPR compliance:</strong> Data retention and deletion policies</li>
                    <li>• <strong>Research ethics:</strong> IRB-compatible data handling</li>
                    <li>• <strong>Export controls:</strong> Appropriate technology transfer compliance</li>
                    <li>• <strong>Institutional policies:</strong> University research data requirements</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
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
                  Review generated certificates, acceptance status, and compare different approaches.
                </p>
                <div className="bg-purple-50 rounded-xl p-4 text-sm">
                  <p className="font-medium text-purple-900 mb-2">What you can analyze:</p>
                  <ul className="text-purple-800 space-y-1 text-xs">
                    <li>• Certificate acceptance status and mathematical validity</li>
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
                  <h4 className="font-medium text-gray-900">Advanced LLM Integration</h4>
                  <p className="text-sm text-gray-600">Claude 4 integration with both direct generation and conversational modes for iterative mathematical reasoning</p>
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
                  <h4 className="font-medium text-gray-900">Real-time Acceptance</h4>
                  <p className="text-sm text-gray-600">Automatic acceptance checking of generated certificate candidates</p>
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
                ongoing research into the application of Large Language Models for formal methods and stability analysis 
                tasks at the University of Colorado Boulder.
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

      {/* System Architecture & Technical Implementation */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader flex items-center">
            <Cog8ToothIcon className="w-6 h-6 mr-3 text-primary-600" />
            System Architecture & Technical Implementation
          </h2>
          <p className="academic-body text-sm">Comprehensive technical infrastructure and computational methods</p>
        </div>
        <div className="card-body">
          <div className="space-y-8">
            
            {/* Core Architecture */}
            <div className="bg-gray-50 rounded-xl p-6">
              <h4 className="text-lg font-semibold text-gray-900 mb-4">Core System Architecture</h4>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="bg-white rounded-lg p-4 border border-gray-200">
                  <h5 className="font-medium text-gray-900 mb-3 flex items-center">
                    <div className="w-6 h-6 rounded bg-blue-100 flex items-center justify-center mr-2">
                      <span className="text-blue-600 text-xs font-bold">UI</span>
                    </div>
                    Frontend Layer
                  </h5>
                  <ul className="text-gray-700 text-sm space-y-1">
                    <li>• <strong>React 18</strong> with TypeScript</li>
                    <li>• <strong>Material Design 3</strong> + Tailwind CSS</li>
                    <li>• <strong>React Hook Form</strong> + Zod validation</li>
                    <li>• <strong>TanStack Query</strong> for state management</li>
                    <li>• <strong>Vite</strong> for optimized builds</li>
                  </ul>
                </div>
                
                <div className="bg-white rounded-lg p-4 border border-gray-200">
                  <h5 className="font-medium text-gray-900 mb-3 flex items-center">
                    <div className="w-6 h-6 rounded bg-green-100 flex items-center justify-center mr-2">
                      <span className="text-green-600 text-xs font-bold">API</span>
                    </div>
                    Backend Services
                  </h5>
                  <ul className="text-gray-700 text-sm space-y-1">
                    <li>• <strong>Node.js</strong> + Express.js</li>
                    <li>• <strong>AcceptanceService</strong> for validation</li>
                    <li>• <strong>MathService</strong> for computation</li>
                    <li>• <strong>LLMService</strong> for AI integration</li>
                    <li>• <strong>BaselineService</strong> for comparison</li>
                  </ul>
                </div>
                
                <div className="bg-white rounded-lg p-4 border border-gray-200">
                  <h5 className="font-medium text-gray-900 mb-3 flex items-center">
                    <div className="w-6 h-6 rounded bg-purple-100 flex items-center justify-center mr-2">
                      <span className="text-purple-600 text-xs font-bold">DB</span>
                    </div>
                    Data Layer
                  </h5>
                  <ul className="text-gray-700 text-sm space-y-1">
                    <li>• <strong>Firebase/Firestore</strong> (current MVP implementation)</li>
                    <li>• <strong>PostgreSQL</strong> migration planned for production</li>
                    <li>• <strong>JWT</strong> authentication with Firebase Auth</li>
                    <li>• <strong>Google Cloud Storage</strong> for artifacts</li>
                    <li>• <strong>Complete audit logging</strong> for research provenance</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Mathematical Computation */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
                <h4 className="text-lg font-semibold text-blue-900 mb-4 flex items-center">
                  <CalculatorIcon className="w-6 h-6 mr-3 text-blue-600" />
                  Mathematical Computation Engine
                </h4>
                <div className="space-y-4">
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-blue-900 mb-2">Symbolic Mathematics</h5>
                    <ul className="text-blue-800 text-sm space-y-1">
                      <li>• <strong>Computer Algebra System (CAS)</strong> integration</li>
                      <li>• <strong>Expression parsing</strong> and canonicalization</li>
                      <li>• <strong>Automatic differentiation</strong> for gradient computation</li>
                      <li>• <strong>Polynomial manipulation</strong> and simplification</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-blue-900 mb-2">Numerical Analysis</h5>
                    <ul className="text-blue-800 text-sm space-y-1">
                      <li>• <strong>Sobol/LHS sampling</strong> for domain coverage</li>
                      <li>• <strong>Adaptive refinement</strong> algorithms</li>
                      <li>• <strong>Monte-Carlo integration</strong> for volume estimation</li>
                      <li>• <strong>Adversarial optimization</strong> for counterexamples</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 rounded-xl p-6 border border-green-200">
                <h4 className="text-lg font-semibold text-green-900 mb-4 flex items-center">
                  <ShieldCheckIcon className="w-6 h-6 mr-3 text-green-600" />
                  Formal Methods Integration
                </h4>
                <div className="space-y-4">
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-green-900 mb-2">Optimization Solvers</h5>
                    <ul className="text-green-800 text-sm space-y-1">
                      <li>• <strong>SeDuMi/MOSEK</strong> for SDP problems</li>
                      <li>• <strong>CVXOPT</strong> for convex optimization</li>
                      <li>• <strong>YALMIP/CVX</strong> modeling languages</li>
                      <li>• <strong>SOSTOOLS</strong> for SOS decomposition</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-green-900 mb-2">Verification Tools</h5>
                    <ul className="text-green-800 text-sm space-y-1">
                      <li>• <strong>dReal</strong> for δ-complete SMT solving</li>
                      <li>• <strong>Z3</strong> for general SMT constraints</li>
                      <li>• <strong>Flow*/CORA</strong> reachability analysis</li>
                      <li>• <strong>SpaceEx</strong> hybrid system verification</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* LLM Integration */}
            <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 border border-indigo-200">
              <h4 className="text-lg font-semibold text-indigo-900 mb-4 flex items-center">
                <CpuChipIcon className="w-6 h-6 mr-3 text-indigo-600" />
                Large Language Model Integration
              </h4>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-medium text-indigo-900 mb-2">Model Interface</h5>
                  <ul className="text-indigo-800 text-sm space-y-1">
                    <li>• <strong>Anthropic Claude 4</strong> API integration</li>
                    <li>• <strong>Structured prompting</strong> with system descriptions</li>
                    <li>• <strong>JSON schema validation</strong> for responses</li>
                    <li>• <strong>Temperature control</strong> and reproducibility</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-medium text-indigo-900 mb-2">Generation Modes</h5>
                  <ul className="text-indigo-800 text-sm space-y-1">
                    <li>• <strong>Direct expression:</strong> Complete mathematical functions</li>
                    <li>• <strong>Basis + coefficients:</strong> Structured decomposition</li>
                    <li>• <strong>Structure + constraints:</strong> Template-based approach</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-medium text-indigo-900 mb-2">Quality Control</h5>
                  <ul className="text-indigo-800 text-sm space-y-1">
                    <li>• <strong>Syntax validation:</strong> Mathematical parsing</li>
                    <li>• <strong>Semantic checks:</strong> Dimensional consistency</li>
                    <li>• <strong>Rejection handling:</strong> Malformed responses</li>
                    <li>• <strong>Budget management:</strong> Token and call limits</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Deployment & Infrastructure */}
            <div className="bg-gray-50 rounded-xl p-6">
              <h4 className="text-lg font-semibold text-gray-900 mb-4">Cloud Deployment & Infrastructure</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h5 className="font-medium text-gray-900 mb-3">Google Cloud Platform</h5>
                  <ul className="text-gray-700 text-sm space-y-1">
                    <li>• <strong>Cloud Run:</strong> Serverless container deployment</li>
                    <li>• <strong>Cloud SQL:</strong> Managed PostgreSQL instances</li>
                    <li>• <strong>Cloud Storage:</strong> Artifact and result persistence</li>
                    <li>• <strong>Secret Manager:</strong> API key and credential management</li>
                    <li>• <strong>Load Balancer:</strong> Global HTTPS with managed certificates</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-gray-900 mb-3">DevOps & Monitoring</h5>
                  <ul className="text-gray-700 text-sm space-y-1">
                    <li>• <strong>Docker:</strong> Containerized microservices</li>
                    <li>• <strong>Cloud Build:</strong> CI/CD pipeline automation</li>
                    <li>• <strong>Cloud Logging:</strong> Centralized log aggregation</li>
                    <li>• <strong>Error Reporting:</strong> Real-time error monitoring</li>
                    <li>• <strong>Terraform:</strong> Infrastructure as code</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Getting Started */}
      <div className="cu-gradient-light border border-primary-200 rounded-2xl p-6">
        <div className="text-center">
          <h3 className="text-lg font-semibold text-primary-900 mb-2">Ready to Get Started?</h3>
          <p className="text-primary-800 text-sm mb-4">
            Begin by creating your first system specification and exploring the formal methods and stability analysis capabilities.
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
