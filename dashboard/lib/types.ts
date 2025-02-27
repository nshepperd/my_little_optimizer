// types/api.ts

export type OptimizationObjective = 'min' | 'max';

export interface SweepParameterType {
  min: number;
  max: number;
  type: 'linear' | 'log' | 'logit';
}

export interface Project {
  id: string;
  name: string;
  created_at: number;
  sweep_count: number;
}

export interface Sweep {
  id: string;
  name: string;
  parameters: Record<string, SweepParameterType>;
  status: string;
  created_at: number;
  objective: OptimizationObjective;
  num_trials: number;
  project_id?: string;
  project_name?: string;
}

export interface TrialResult {
  id: string;
  parameters: Record<string, number>;
  value: number;
  status: string;
  created_at: number;
  completed_at: number | null;
  trial_number: number;
}

// Request/Response types
export interface ProjectCreateRequest {
  name: string;
}

export interface SweepCreateRequest {
  name: string;
  parameters: Record<string, {
    min: number;
    max: number;
    type: 'linear' | 'log' | 'logit';
  }>;
  objective?: OptimizationObjective;
  project_id?: string;
  project_name?: string;
}

export interface SweepUpdateRequest {
  name?: string;
}

export interface TrialCreateRequest {
  parameters: Record<string, number>;
  value?: number;
}

export interface TrialReportRequest {
  value: number;
}

export interface SliceVisualizationRequest {
  param_name: string;
}

export interface SliceVisualizationDatapoint {
  x: number;
  y_mean: number;
  y_std: number;
  params: Record<string, number>;
}

export interface SliceVisualization {
  sweep_id: string;
  param_name: string;
  computed_at: number;
  data: SliceVisualizationDatapoint[];
}

export interface SliceVisualizationResponse {
  cached: SliceVisualization | null;
  job_id: string | null;
}

export interface PollResponse<T> {
  status: 'done' | 'error' | 'pending' | 'running';
  result: T | null;
  message: string | null;
}

export interface ApiResponse<T> {
  status: 'ok' | 'error';
  data: T;
  message: string;
  code: string;
}