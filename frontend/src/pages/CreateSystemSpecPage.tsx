import { useState, useEffect } from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { ChevronLeftIcon, ChevronRightIcon, PlusIcon, TrashIcon } from '@heroicons/react/24/outline';
import { clsx } from 'clsx';

import { SystemSpecRequest } from '@/types/api';
import { api } from '@/services/api';

// Form data type for manual validation
interface SystemSpecForm {
  name: string;
  description?: string;
  system_type: 'continuous' | 'discrete' | 'hybrid';
  dimension: number;
  dynamics: {
    type: 'polynomial' | 'nonlinear' | 'linear' | 'piecewise';
    variables: string[];
    equations: string[];
    domain?: {
      bounds?: Record<string, { min?: number; max?: number }>;
      constraints?: string[];
    };
  };
  constraints?: any;
  initial_set?: any;
  unsafe_set?: any;
}

const STEPS = [
  { id: 'basic', title: 'Basic Information' },
  { id: 'dynamics', title: 'System Dynamics' },
  { id: 'sets', title: 'Sets & Constraints' },
  { id: 'review', title: 'Review & Create' },
];

export default function CreateSystemSpecPage() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);

  const form = useForm<SystemSpecForm>({
    // No resolver - we use manual validation only
    defaultValues: {
      name: '',
      description: '',
      system_type: 'continuous',
      dimension: 2,
      dynamics: {
        type: 'polynomial',
        variables: ['x1', 'x2'],
        equations: ['', ''],
        domain: {
          bounds: {},
          constraints: [],
        },
      },
      constraints: '',
      initial_set: '',
      unsafe_set: '',
    },
    mode: 'onChange', // Minimal form state updates
  });

  const { fields: variableFields, append: appendVariable, remove: removeVariable } = useFieldArray({
    control: form.control,
    name: 'dynamics.variables',
  });

  const { append: appendEquation, remove: removeEquation } = useFieldArray({
    control: form.control,
    name: 'dynamics.equations',
  });

  // Initialize equation fields to match variable fields
  useEffect(() => {
    const currentVariables = form.getValues('dynamics.variables') || [];
    const currentEquations = form.getValues('dynamics.equations') || [];
    
    // If equations array is shorter than variables array, add missing equations
    if (currentEquations.length < currentVariables.length) {
      const missingCount = currentVariables.length - currentEquations.length;
      for (let i = 0; i < missingCount; i++) {
        appendEquation('');
      }
    }
  }, [form, appendEquation]);

  const createSystemSpecMutation = useMutation({
    mutationFn: async (data: SystemSpecRequest) => {
      return await api.createSystemSpec(data);
    },
    onSuccess: () => {
      toast.success('System specification created successfully!');
      navigate('/system-specs');
    },
    onError: (error: any) => {
      toast.error(error?.response?.data?.error || 'Failed to create system specification');
    },
  });

  const onSubmit = async (data: SystemSpecForm) => {
    // Basic validation only - detailed validation already done step-by-step
    if (!data.name || !data.system_type || !data.dimension || !data.dynamics?.type) {
      toast.error('Please complete all required fields');
      return;
    }

    // Parse JSON strings for sets before submitting
    const processedData = {
      ...data,
      initial_set: data.initial_set ? (() => {
        try {
          return JSON.parse(data.initial_set);
        } catch {
          return data.initial_set;
        }
      })() : undefined,
      unsafe_set: data.unsafe_set ? (() => {
        try {
          return JSON.parse(data.unsafe_set);
        } catch {
          return data.unsafe_set;
        }
      })() : undefined,
      constraints: data.constraints ? (() => {
        try {
          return JSON.parse(data.constraints);
        } catch {
          return data.constraints;
        }
      })() : undefined,
    };

    createSystemSpecMutation.mutate(processedData);
  };

  const nextStep = async () => {
    // Manual validation for each step to avoid schema refinement issues
    const formData = form.getValues();
    let validationErrors: string[] = [];

    switch (currentStep) {
      case 0: // Basic Information
        if (!formData.name || formData.name.trim() === '') {
          validationErrors.push('System name is required');
        }
        if (formData.name && formData.name.length > 255) {
          validationErrors.push('Name is too long (maximum 255 characters)');
        }
        if (formData.name && !/^[a-zA-Z0-9\s\-_()]+$/.test(formData.name)) {
          validationErrors.push('Name can only contain letters, numbers, spaces, hyphens, underscores, and parentheses');
        }
        if (!formData.system_type) {
          validationErrors.push('Please select a system type');
        }
        if (!formData.dimension || formData.dimension < 1 || formData.dimension > 20) {
          validationErrors.push('Dimension must be between 1 and 20');
        }
        if (!formData.dynamics?.type) {
          validationErrors.push('Please select a dynamics type');
        }
        break;

      case 1: // System Dynamics
        if (!formData.dynamics?.variables || formData.dynamics.variables.length === 0) {
          validationErrors.push('At least one variable is required');
        }
        if (formData.dynamics?.variables) {
          formData.dynamics.variables.forEach((variable, index) => {
            if (!variable || variable.trim() === '') {
              validationErrors.push(`Variable ${index + 1} name cannot be empty`);
            }
            if (variable && !/^[a-zA-Z][a-zA-Z0-9_]*$/.test(variable)) {
              validationErrors.push(`Variable "${variable}" must start with a letter and contain only letters, numbers, and underscores`);
            }
          });
          
          // Check for duplicate variable names
          const uniqueVars = new Set(formData.dynamics.variables);
          if (uniqueVars.size !== formData.dynamics.variables.length) {
            validationErrors.push('Variable names must be unique');
          }
        }
        if (formData.dynamics?.equations) {
          formData.dynamics.equations.forEach((equation, index) => {
            if (!equation || equation.trim() === '') {
              validationErrors.push(`Equation ${index + 1} cannot be empty - please define the differential equation`);
            }
            if (equation && equation.length > 500) {
              validationErrors.push(`Equation ${index + 1} is too long (maximum 500 characters)`);
            }
          });
        }
        if (formData.dynamics?.variables && formData.dynamics?.equations) {
          if (formData.dynamics.variables.length !== formData.dynamics.equations.length) {
            validationErrors.push('Number of variables must match number of equations');
          }
        }
        break;

      case 2: // Sets & Constraints - minimal validation since these are optional
        // Optional validation could be added here if needed
        break;
    }

    if (validationErrors.length > 0) {
      toast.error(`Please fix these issues: ${validationErrors.join('; ')}`);
      return;
    }
    
    if (currentStep < STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const updateDimension = (newDimension: number) => {
    const currentVars = form.getValues('dynamics.variables');
    const currentEqs = form.getValues('dynamics.equations');
    
    if (newDimension > currentVars.length) {
      // Add variables and equations
      const varsToAdd = newDimension - currentVars.length;
      for (let i = 0; i < varsToAdd; i++) {
        appendVariable(`x${currentVars.length + i + 1}`);
        appendEquation('');
      }
    } else if (newDimension < currentVars.length) {
      // Remove variables and equations
      const varsToRemove = currentVars.length - newDimension;
      for (let i = 0; i < varsToRemove; i++) {
        removeVariable(currentVars.length - 1 - i);
        removeEquation(currentEqs.length - 1 - i);
      }
    }
    
    form.setValue('dimension', newDimension);
  };



  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Create System Specification</h1>
        <p className="mt-2 text-gray-600">
          Define a new dynamical system for formal verification analysis.
        </p>
      </div>

      {/* Progress Steps */}
      <div className="flex items-center justify-between">
        {STEPS.map((step, index) => (
          <div key={step.id} className="flex items-center">
            <div
              className={clsx(
                'flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium',
                index <= currentStep
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-600'
              )}
            >
              {index + 1}
            </div>
            <div className="ml-3">
              <p className={clsx('text-sm font-medium', index <= currentStep ? 'text-blue-600' : 'text-gray-500')}>
                {step.title}
              </p>
            </div>
            {index < STEPS.length - 1 && (
              <div className={clsx('ml-6 w-16 h-0.5', index < currentStep ? 'bg-blue-600' : 'bg-gray-200')} />
            )}
          </div>
        ))}
      </div>

      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">

      <div className="card">
        <div className="card-body">
            {/* Step 1: Basic Information */}
            {currentStep === 0 && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-900">Basic Information</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Name *
                    </label>
                    <input
                      type="text"
                      {...form.register('name')}
                      className="input"
                      placeholder="e.g., Van der Pol Oscillator"
                    />

                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      System Type *
                    </label>
                    <select {...form.register('system_type')} className="input">
                      <option value="continuous">Continuous</option>
                      <option value="discrete">Discrete</option>
                      <option value="hybrid">Hybrid</option>
                    </select>

                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Dimension *
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="20"
                      {...form.register('dimension', { 
                        valueAsNumber: true,
                        onChange: (e) => updateDimension(parseInt(e.target.value) || 1)
                      })}
                      className="input"
                    />

                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Dynamics Type *
                    </label>
                    <select {...form.register('dynamics.type')} className="input">
                      <option value="linear">Linear</option>
                      <option value="polynomial">Polynomial</option>
                      <option value="nonlinear">Nonlinear</option>
                      <option value="piecewise">Piecewise</option>
                    </select>

                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Description
                  </label>
                  <textarea
                    {...form.register('description')}
                    rows={3}
                    className="input"
                    placeholder="Optional description of the system..."
                  />
                </div>
              </div>
            )}

            {/* Step 2: System Dynamics */}
            {currentStep === 1 && (
              <div className="space-y-8">
                <div>
                  <h2 className="academic-subheader">System Dynamics</h2>
                  <p className="academic-body text-sm">
                    Define the state variables and their corresponding differential equations.
                  </p>
                </div>
                
                {/* Variables Section */}
                <div className="surface-elevated p-6">
                  <div className="flex items-center justify-between mb-4">
                    <label className="text-lg font-medium text-gray-900">
                      State Variables
                    </label>
                    <button
                      type="button"
                      onClick={() => {
                        appendVariable(`x${variableFields.length + 1}`);
                        appendEquation('');
                        form.setValue('dimension', variableFields.length + 1);
                      }}
                      className="btn-secondary text-sm"
                    >
                      <PlusIcon className="w-4 h-4 mr-2" />
                      Add Variable
                    </button>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                    {variableFields.map((field, index) => (
                      <div key={field.id} className="flex items-center gap-2">
                        <div className="flex-1">
                          <label className="block text-xs font-medium text-gray-600 mb-1">
                            Variable {index + 1}
                          </label>
                          <input
                            {...form.register(`dynamics.variables.${index}`)}
                            className="input w-full"
                            placeholder={`x${index + 1}`}
                          />
                        </div>
                        {variableFields.length > 1 && (
                          <button
                            type="button"
                            onClick={() => {
                              removeVariable(index);
                              removeEquation(index);
                              form.setValue('dimension', variableFields.length - 1);
                            }}
                            className="mt-6 p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                            title="Remove variable"
                          >
                            <TrashIcon className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Equations Section */}
                <div className="surface-elevated p-6">
                  <div className="mb-4">
                    <label className="text-lg font-medium text-gray-900 block">
                      Differential Equations
                    </label>
                    <p className="text-sm text-gray-600 mt-1">
                      Define dx/dt for each state variable
                    </p>
                  </div>
                  <div className="space-y-4">
                    {variableFields.length > 0 ? (
                      variableFields.map((_, index) => {
                        const varName = form.watch('dynamics.variables')?.[index] || `x${index + 1}`;
                        return (
                          <div key={`equation-${index}`} className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">
                              d{varName}/dt =
                            </label>
                            <div className="relative">
                                                          <input
                              {...form.register(`dynamics.equations.${index}`)}
                              className="input w-full font-mono text-sm"
                              placeholder={`Enter equation for d${varName}/dt (e.g., x2, -sin(x1), x1^2 + x2)`}
                            />
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div className="text-sm text-gray-500 bg-yellow-50 p-4 rounded">
                        No variables defined yet. Please add variables first.
                      </div>
                    )}
                  </div>
                </div>

                {/* Domain Constraints Section */}
                <div className="surface-elevated p-6">
                  <div className="mb-4">
                    <label className="text-lg font-medium text-gray-900 block">
                      Domain Constraints
                    </label>
                    <p className="text-sm text-gray-600 mt-1">
                      Define the operating region and bounds for your system variables
                    </p>
                  </div>
                  
                  {/* Variable Bounds */}
                  <div className="space-y-4 mb-6">
                    <h4 className="text-sm font-medium text-gray-700">Variable Bounds (Optional)</h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {variableFields.map((_, index) => {
                        const varName = form.watch('dynamics.variables')?.[index] || `x${index + 1}`;
                        return (
                          <div key={`bounds-${index}`} className="space-y-2">
                            <label className="block text-xs font-medium text-gray-600">
                              {varName} bounds
                            </label>
                            <div className="flex items-center gap-2">
                              <input
                                {...form.register(`dynamics.domain.bounds.${varName}.min`, { valueAsNumber: true })}
                                type="number"
                                step="any"
                                className="input flex-1 text-sm"
                                placeholder="min"
                              />
                              <span className="text-gray-400">&le; {varName} &le;</span>
                              <input
                                {...form.register(`dynamics.domain.bounds.${varName}.max`, { valueAsNumber: true })}
                                type="number"
                                step="any"
                                className="input flex-1 text-sm"
                                placeholder="max"
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* General Constraints */}
                  <div className="space-y-4">
                    <h4 className="text-sm font-medium text-gray-700">General Constraints (Optional)</h4>
                    <textarea
                      {...form.register('dynamics.domain.constraints.0')}
                      rows={2}
                      className="input w-full font-mono text-sm"
                      placeholder="e.g., x1^2 + x2^2 ≤ 4, x1 ≥ 0, |x2| ≤ 2"
                    />
                    <p className="text-xs text-gray-500">
                      Enter mathematical constraints separated by commas. Use standard mathematical notation.
                    </p>
                  </div>
                </div>

                {/* Examples Section */}
                <div className="cu-gradient-light border border-primary-200 rounded-2xl p-6">
                  <h3 className="text-sm font-semibold text-primary-900 mb-3 flex items-center">
                    <span className="w-2 h-2 bg-primary-600 rounded-full mr-2"></span>
                    Examples
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="font-medium text-primary-800 mb-1">Equations:</p>
                      <p className="text-primary-700 font-mono">x2</p>
                      <p className="text-primary-700 font-mono">-sin(x1)</p>
                      <p className="text-primary-700 font-mono">x1^2 - x2</p>
                    </div>
                    <div>
                      <p className="font-medium text-primary-800 mb-1">Variable Bounds:</p>
                      <p className="text-primary-700 font-mono">x1: [-π, π]</p>
                      <p className="text-primary-700 font-mono">x2: [-5, 5]</p>
                    </div>
                    <div>
                      <p className="font-medium text-primary-800 mb-1">Domain Constraints:</p>
                      <p className="text-primary-700 font-mono">x1^2 + x2^2 &le; 4</p>
                      <p className="text-primary-700 font-mono">x1 &ge; 0</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Step 3: Sets & Constraints */}
            {currentStep === 2 && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-900">Sets & Constraints</h2>
                <p className="text-gray-600 text-sm">
                  Define the initial set, unsafe set, and domain constraints for your system.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Initial Set (Optional)
                    </label>
                    <textarea
                      {...form.register('initial_set')}
                      value={typeof form.watch('initial_set') === 'object' ? JSON.stringify(form.watch('initial_set'), null, 2) : form.watch('initial_set') || ''}
                      onChange={(e) => form.setValue('initial_set', e.target.value)}
                      rows={3}
                      className="input"
                      placeholder='{"type": "box", "bounds": {"x1": [-1, 1], "x2": [-1, 1]}}'
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Unsafe Set (Optional)
                    </label>
                    <textarea
                      {...form.register('unsafe_set')}
                      value={typeof form.watch('unsafe_set') === 'object' ? JSON.stringify(form.watch('unsafe_set'), null, 2) : form.watch('unsafe_set') || ''}
                      onChange={(e) => form.setValue('unsafe_set', e.target.value)}
                      rows={3}
                      className="input"
                      placeholder='{"type": "ball", "center": [2, 2], "radius": 0.5}'
                    />
                  </div>
                </div>

                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-yellow-900 mb-2">Set Definition Examples:</h3>
                  <div className="text-sm text-yellow-700 space-y-1">
                    <p><strong>Box:</strong> {`{"type": "box", "bounds": {"x1": [-1, 1], "x2": [-1, 1]}}`}</p>
                    <p><strong>Ball:</strong> {`{"type": "ball", "center": [0, 0], "radius": 1}`}</p>
                    <p><strong>Polynomial:</strong> {`{"type": "polynomial", "expression": "x1^2 + x2^2 <= 1"}`}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Step 4: Review */}
            {currentStep === 3 && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-900">Review & Create</h2>
                
                <div className="bg-gray-50 rounded-lg p-6 space-y-4">
                  <div>
                    <h3 className="font-medium text-gray-900">System Information</h3>
                    <dl className="mt-2 grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <dt className="text-gray-500">Name:</dt>
                        <dd className="text-gray-900">{form.watch('name')}</dd>
                      </div>
                      <div>
                        <dt className="text-gray-500">Type:</dt>
                        <dd className="text-gray-900 capitalize">{form.watch('system_type')}</dd>
                      </div>
                      <div>
                        <dt className="text-gray-500">Dimension:</dt>
                        <dd className="text-gray-900">{form.watch('dimension')}</dd>
                      </div>
                      <div>
                        <dt className="text-gray-500">Dynamics Type:</dt>
                        <dd className="text-gray-900 capitalize">{form.watch('dynamics.type')}</dd>
                      </div>
                    </dl>
                  </div>

                  <div>
                    <h3 className="font-medium text-gray-900">Variables & Equations</h3>
                    <div className="mt-2 space-y-1">
                      {form.watch('dynamics.variables')?.map((variable, index) => (
                        <div key={index} className="text-sm text-gray-600">
                          <code>d{variable}/dt = {form.watch('dynamics.equations')?.[index] || '(empty)'}</code>
                        </div>
                      ))}
                    </div>
                  </div>

                  {form.watch('description') && (
                    <div>
                      <h3 className="font-medium text-gray-900">Description</h3>
                      <p className="mt-1 text-sm text-gray-600">{form.watch('description')}</p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Navigation */}
        <div className="flex justify-between">
          <button
            type="button"
            onClick={prevStep}
            disabled={currentStep === 0}
            className={clsx(
              'flex items-center px-4 py-2 text-sm font-medium rounded-lg',
              currentStep === 0
                ? 'text-gray-400 cursor-not-allowed'
                : 'text-gray-700 bg-white border border-gray-300 hover:bg-gray-50'
            )}
          >
            <ChevronLeftIcon className="w-4 h-4 mr-1" />
            Previous
          </button>

          {currentStep < STEPS.length - 1 ? (
            <button
              type="button"
              onClick={nextStep}
              className="flex items-center px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
            >
              Next
              <ChevronRightIcon className="w-4 h-4 ml-1" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={createSystemSpecMutation.isPending}
              className="flex items-center px-6 py-2 text-sm font-medium text-white bg-green-600 rounded-lg hover:bg-green-700 disabled:opacity-50"
            >
              {createSystemSpecMutation.isPending ? 'Creating...' : 'Create System Specification'}
            </button>
          )}
      </div>
      </form>
    </div>
  );
}
