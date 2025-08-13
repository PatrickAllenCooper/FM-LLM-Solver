import { useState } from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { ChevronLeftIcon, ChevronRightIcon, PlusIcon, TrashIcon } from '@heroicons/react/24/outline';
import { clsx } from 'clsx';

import { SystemSpecRequest } from '@/types/api';
import { api } from '@/services/api';

// Validation schema
const SystemSpecSchema = z.object({
  name: z.string().min(1, 'Name is required').max(255, 'Name too long'),
  description: z.string().optional(),
  system_type: z.enum(['continuous', 'discrete', 'hybrid']),
  dimension: z.number().min(1, 'Dimension must be at least 1').max(20, 'Dimension cannot exceed 20'),
  dynamics: z.object({
    type: z.enum(['polynomial', 'nonlinear', 'linear', 'piecewise']),
    variables: z.array(z.string().min(1, 'Variable name required')),
    equations: z.array(z.string().min(1, 'Equation required')),
    domain: z.object({
      bounds: z.record(z.object({
        min: z.number().optional(),
        max: z.number().optional(),
      })).optional(),
      constraints: z.array(z.string()).optional(),
    }).optional(),
  }),
  constraints: z.any().optional(),
  initial_set: z.any().optional(),
  unsafe_set: z.any().optional(),
});

type SystemSpecForm = z.infer<typeof SystemSpecSchema>;

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
    resolver: zodResolver(SystemSpecSchema),
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
      constraints: {},
      initial_set: {},
      unsafe_set: {},
    },
    mode: 'onChange',
  });

  const { fields: variableFields, append: appendVariable, remove: removeVariable } = useFieldArray({
    control: form.control,
    name: 'dynamics.variables',
  });

  const { fields: equationFields, append: appendEquation, remove: removeEquation } = useFieldArray({
    control: form.control,
    name: 'dynamics.equations',
  });

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

  const onSubmit = (data: SystemSpecForm) => {
    createSystemSpecMutation.mutate(data);
  };

  const nextStep = () => {
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
                    {form.formState.errors.name && (
                      <p className="mt-1 text-sm text-red-600">{form.formState.errors.name.message}</p>
                    )}
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
                    {form.formState.errors.dimension && (
                      <p className="mt-1 text-sm text-red-600">{form.formState.errors.dimension.message}</p>
                    )}
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
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-900">System Dynamics</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Variables */}
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <label className="block text-sm font-medium text-gray-700">
                        State Variables
                      </label>
                      <button
                        type="button"
                        onClick={() => {
                          appendVariable(`x${variableFields.length + 1}`);
                          appendEquation('');
                          form.setValue('dimension', variableFields.length + 1);
                        }}
                        className="btn btn-sm btn-outline"
                      >
                        <PlusIcon className="w-4 h-4 mr-1" />
                        Add Variable
                      </button>
                    </div>
                    <div className="space-y-2">
                      {variableFields.map((field, index) => (
                        <div key={field.id} className="flex items-center gap-2">
                          <input
                            {...form.register(`dynamics.variables.${index}`)}
                            className="input flex-1"
                            placeholder={`x${index + 1}`}
                          />
                          {variableFields.length > 1 && (
                            <button
                              type="button"
                              onClick={() => {
                                removeVariable(index);
                                removeEquation(index);
                                form.setValue('dimension', variableFields.length - 1);
                              }}
                              className="btn btn-sm btn-outline text-red-600 hover:bg-red-50"
                            >
                              <TrashIcon className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Equations */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                      Dynamic Equations
                    </label>
                    <div className="space-y-2">
                      {equationFields.map((field, index) => (
                        <div key={field.id} className="flex items-center gap-2">
                          <span className="text-sm text-gray-500 w-8">
                            d{form.watch('dynamics.variables')?.[index] || `x${index + 1}`}/dt =
                          </span>
                          <input
                            {...form.register(`dynamics.equations.${index}`)}
                            className="input flex-1"
                            placeholder="e.g., x2"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-blue-900 mb-2">Examples:</h3>
                  <div className="text-sm text-blue-700 space-y-1">
                    <p><strong>Linear:</strong> -x1 + 2*x2</p>
                    <p><strong>Polynomial:</strong> x1^2 - x1*x2 + x2</p>
                    <p><strong>Nonlinear:</strong> sin(x1) - cos(x2)*x1</p>
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
                      {...form.register('initial_set', {
                        setValueAs: (value) => {
                          try {
                            return value ? JSON.parse(value) : undefined;
                          } catch {
                            return value;
                          }
                        }
                      })}
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
                      {...form.register('unsafe_set', {
                        setValueAs: (value) => {
                          try {
                            return value ? JSON.parse(value) : undefined;
                          } catch {
                            return value;
                          }
                        }
                      })}
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
