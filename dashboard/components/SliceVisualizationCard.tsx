import React, { useState, useEffect } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  TooltipProps,
} from "recharts";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Sweep, SliceVisualization, SliceVisualizationResponse, ApiResponse, PollResponse } from "@/lib/types";
import { useInterval } from "@/lib/useInterval";

interface SliceVisualizationCardProps {
  sweep: Sweep;
  autoRefresh: boolean;
}

const POLL_INTERVAL = 1000; // 1 second for job polling
const REFRESH_INTERVAL = 5000; // 5 seconds for auto-refresh

const SliceVisualizationCard = ({ sweep, autoRefresh }: SliceVisualizationCardProps) => {
  const [selectedParam, setSelectedParam] = useState<string | null>(null);
  const [visualization, setVisualization] = useState<SliceVisualization | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedParam) {
      const firstParam = Object.keys(sweep.parameters)[0];
      if (firstParam) {
        setSelectedParam(firstParam);
      }
    }
  }, [sweep.parameters]);

  const fetchVisualization = async () => {
    if (!selectedParam) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(
        `/api/sweeps/${sweep.id}/visualize/slice`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ param_name: selectedParam })
        }
      );
      const result: ApiResponse<SliceVisualizationResponse> = await response.json();
      
      if (result.status === 'ok') {
        const resp = result.data;
        if (resp.cached) {
          setVisualization(resp.cached);
          setLoading(false);
        }
        if (resp.job_id) {
          setJobId(resp.job_id);
        }
      } else {
        setError(result.message || 'Failed to fetch visualization');
        setLoading(false);
      }
    } catch (err) {
      setError('Failed to fetch visualization');
      setLoading(false);
    }
  };

  // Poll for job results
  useInterval(async () => {
    if (!jobId) return;

    try {
      const response = await fetch(`/api/poll/${jobId}`, {method: 'POST'});
      const result: ApiResponse<PollResponse<SliceVisualization>> = await response.json();

      if (result.status === 'ok') {
        if (result.data.status === 'done') {
            setVisualization(result.data.result);
            setJobId(null);
        } else if (result.data.status === 'error') {
            setError(result.data.message);
            setJobId(null);
        }
      } else if (result.status === 'error') {
        setError(result.message);
        setJobId(null);
      }
    } catch (err) {
      setError('Failed to poll for results');
    //   setJobId(null);
    }
  }, jobId ? POLL_INTERVAL : null);

  // Auto-refresh
  useInterval(() => {
    fetchVisualization();
  }, autoRefresh ? REFRESH_INTERVAL : null);

  // Initial fetch
  useEffect(() => {
    if (visualization && sweep.id !== visualization.sweep_id) {
        setVisualization(null);
    }
    if (visualization && selectedParam !== visualization.param_name) {
        setVisualization(null);
    }
    if (selectedParam) {
      fetchVisualization();
    }
  }, [sweep.id, selectedParam]);

  if (!selectedParam) return null;

  const renderTooltipWithoutRange = ({ payload, content, ...rest }: TooltipProps<number, string>) => {
    const newPayload = payload?.filter((x) => x.dataKey !== "a") ?? [];
    return <Tooltip payload={newPayload} {...rest} />;
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Parameter Slice Visualization</CardTitle>
          <Select
            value={selectedParam}
            onValueChange={setSelectedParam}
          >
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Select parameter" />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(sweep.parameters).map(([name, param]) => (
                <SelectItem key={name} value={name}>
                  {name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent>
        {(loading || jobId) && (
          <div className="absolute right-4 top-4 text-sm text-gray-500">
            {jobId ? "Computing..." : "Loading..."}
          </div>
        )}
        
        {error && (
          <div className="h-64 flex items-center justify-center text-red-500">
            {error}
          </div>
        )}

        {visualization && (
          <>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="x"
                    type="number"
                    domain={['auto', 'auto']}
                    label={{ value: selectedParam, position: "bottom" }}
                  />
                  <YAxis
                    label={{
                      value: "Value",
                      angle: -90,
                      position: "insideLeft",
                    }}
                  />
                  <Tooltip content={renderTooltipWithoutRange} isAnimationActive={false} />
                  <Legend />

                  {/* 3-sigma confidence band */}
                  <Area
                    name="3σ confidence"
                    data={visualization.data.map((d) => ({...d, a: [d.y_mean - 3 * d.y_std, d.y_mean + 3 * d.y_std]}))}
                    dataKey="a"
                    stroke="none"
                    fillOpacity={0.1}
                    fill="#2563eb"
                    activeDot={false}
                    isAnimationActive={false}
                  />

                  {/* 2-sigma confidence band */}
                  <Area
                    name="2σ confidence"
                    data={visualization.data.map((d) => ({...d, a: [d.y_mean - 2 * d.y_std, d.y_mean + 2 * d.y_std]}))}
                    dataKey="a"
                    stroke="none"
                    fillOpacity={0.1}
                    fill="#2563eb"
                    activeDot={false}
                    isAnimationActive={false}
                  />

                  {/* 1-sigma confidence band */}
                  <Area
                    name="1σ confidence"
                    data={visualization.data.map((d) => ({...d, a: [d.y_mean - 1 * d.y_std, d.y_mean + 1 * d.y_std]}))}
                    dataKey="a"
                    stroke="none"
                    fillOpacity={0.1}
                    fill="#2563eb"
                    activeDot={false}
                    isAnimationActive={false}
                  />

                  {/* Mean prediction line */}
                  <Line
                    name="Mean prediction"
                    data={visualization.data}
                    dataKey="y_mean"
                    stroke="#2563eb"
                    dot={false}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="text-sm text-gray-500 mt-2">
              Last computed: {new Date(visualization.computed_at * 1000).toLocaleString()}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default SliceVisualizationCard;