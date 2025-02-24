// types/api.ts

export type OptimizationObjective = 'min' | 'max';

export interface SweepParameterType {
  min: number;
  max: number;
  log: boolean;
}

export interface Sweep {
  id: string;
  name: string;
  parameters: Record<string, SweepParameterType>;
  status: string;
  created_at: number;
  objective: OptimizationObjective;
}

export interface ExperimentResult {
  id: string;
  parameters: Record<string, number>;
  value: number;
  status: string;
  created_at: number;
  completed_at: number | null;
}

// Request/Response types
export interface SweepCreateRequest {
  name: string;
  parameters: Array<{
    name: string;
    min: number;
    max: number;
    log?: boolean;
  }>;
  objective?: OptimizationObjective;
}

export interface SweepUpdateRequest {
  name?: string;
}

export interface ExperimentCreateRequest {
  parameters: Record<string, number>;
  value?: number;
}

export interface ExperimentReportRequest {
  value: number;
}