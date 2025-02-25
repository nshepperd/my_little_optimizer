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
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sweep,
  SliceVisualization,
  SliceVisualizationResponse,
  ApiResponse,
  PollResponse,
  SliceVisualizationDatapoint,
} from "@/lib/types";
import { useInterval } from "@/lib/useInterval";

interface SliceVisualizationCardProps {
  sweep: Sweep;
  autoRefresh: boolean;
}

const POLL_INTERVAL = 1000; // 1 second for job polling
const REFRESH_INTERVAL = 5000; // 5 seconds for auto-refresh

const SliceVisualizationCard = ({
  sweep,
  autoRefresh,
}: SliceVisualizationCardProps) => {
  const [selectedParam, setSelectedParam] = useState<string | null>(null);
  const [visualization, setVisualization] = useState<SliceVisualization | null>(
    null
  );
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
      const response = await fetch(`/api/sweeps/${sweep.id}/visualize/slice`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ param_name: selectedParam }),
      });
      const result: ApiResponse<SliceVisualizationResponse> =
        await response.json();

      if (result.status === "ok") {
        const resp = result.data;
        if (resp.cached) {
          if (
            resp.cached.sweep_id === sweep.id &&
            resp.cached.param_name === selectedParam
          ) {
            setVisualization(resp.cached);
            setLoading(false);
          }
        }
        if (resp.job_id) {
          setJobId(resp.job_id);
        }
      } else if (result.code === "no_data") {
        setError(result.message);
        setLoading(false);
      } else {
        setError(result.message || "Failed to fetch visualization");
      }
    } catch (err) {
      setError("Failed to fetch visualization");
    }
  };

  // Poll for job results
  useInterval(
    async () => {
      if (!jobId) return;

      try {
        const response = await fetch(`/api/poll/${jobId}`, { method: "POST" });
        const result: ApiResponse<PollResponse<SliceVisualization>> =
          await response.json();

        if (result.status === "ok") {
          if (result.data.status === "done") {
            let new_vis = result.data.result;
            if (
              new_vis &&
              new_vis.sweep_id === sweep.id &&
              new_vis.param_name === selectedParam
            ) {
              setVisualization(result.data.result);
              setLoading(false);
            }
            setJobId(null);
          } else if (result.data.status === "error") {
            setError(result.data.message);
            setJobId(null);
          }
        } else if (result.status === "error") {
          setError(result.message);
          setJobId(null);
        }
      } catch (err) {
        setError("Failed to poll for results");
        //   setJobId(null);
      }
    },
    jobId ? POLL_INTERVAL : null
  );

  // Auto-refresh
  useInterval(
    () => {
      if (!jobId) {
        // Only fetch visualization if we're not already waiting for one to come back.
        fetchVisualization();
      }
    },
    autoRefresh ? REFRESH_INTERVAL : null
  );

  useEffect(() => {
    if (selectedParam && !Object.keys(sweep.parameters).includes(selectedParam)) {
      setSelectedParam(Object.keys(sweep.parameters)[0]);
    }
  }, [sweep]);

  // Initial fetch
  useEffect(() => {
    if (selectedParam && Object.keys(sweep.parameters).includes(selectedParam)) {
      fetchVisualization();
    }
  }, [sweep, selectedParam]);

  if (!selectedParam) return null;

  const renderTooltipWithoutRange = ({
    payload,
    content,
    ...rest
  }: TooltipProps<number, string>) => {
    let newPayload = payload?.filter((x) => x.dataKey !== "a") ?? [];

    if (newPayload.length === 0) return null;
    const d: SliceVisualizationDatapoint = newPayload[0].payload;

    let opttable = null;
    if (Object.keys(sweep.parameters).length > 1) {
      const other_params = Object.entries(d.params).filter(
        (kv) => kv[0] !== selectedParam
      );
      opttable = (
        <div className="relative mt-4 pt-1 text-xs">
          <div className="absolute -top-1 left-2 px-1 bg-white text-xs text-gray-700">
            Optimal Params
          </div>
          <div className="border rounded p-2 pt-1">
            <table className="w-full">
              <tbody>
                {other_params.map(([k, v]) => (
                  <tr key={k}>
                    <td className="pr-3 text-gray-600">{k}:</td>
                    <td className="text-right font-mono">{v.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    }

    return (
      <div className="bg-white p-3 border rounded-md shadow-md">
        <p className="font-medium text-gray-800 mb-2">
          {selectedParam}: {d.x.toFixed(4)}
        </p>
        <div className="text-gray-600 mb-3">
          <p>Mean: {d.y_mean.toFixed(4)}</p>
          <p>Std Dev: {d.y_std.toFixed(4)}</p>
        </div>
        {opttable}
      </div>
    );

    // newPayload = newPayload.map((x) => {
    //     console.log(x.payload);
    //     return x; });
    // return <Tooltip payload={newPayload} {...rest} />;
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between">
          <CardTitle>Parameter Slice Visualization</CardTitle>
          {(loading || jobId) && (
            <div className="text-sm text-gray-500">
              {jobId ? "Computing..." : "Loading..."}
            </div>
          )}
          {error && (
            <div className="flex items-center justify-center text-red-500">
              {error}
            </div>
          )}
          <Select value={selectedParam} onValueChange={setSelectedParam}>
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
        {visualization &&
          selectedParam && Object.keys(sweep.parameters).includes(selectedParam) &&
          visualization.sweep_id === sweep.id && (
            <>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    margin={{ top: 5, right: 30, left: 20, bottom: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="x"
                      type="number"
                      domain={["auto", "auto"]}
                      label={{ value: selectedParam, position: "bottom" }}
                      scale={
                        sweep.parameters[selectedParam].log ? "log" : "linear"
                      }
                      ticks={(() => {
                        // Only generate custom ticks for log scale
                        if (!sweep.parameters[selectedParam]?.log)
                          return undefined;

                        // Get parameter range
                        const min = sweep.parameters[selectedParam].min;
                        const max = sweep.parameters[selectedParam].max;

                        // Generate logarithmic tick positions
                        const logTicks = [];

                        // Find appropriate decade bounds
                        const minDecade = Math.floor(Math.log10(min));
                        const maxDecade = Math.ceil(Math.log10(max));

                        // For each decade
                        for (
                          let decade = minDecade;
                          decade <= maxDecade;
                          decade++
                        ) {
                          // Add 1, 2, 5 for each decade (standard log scale divisions)
                          for (const multiplier of [1, 2, 5]) {
                            const tickValue = multiplier * Math.pow(10, decade);
                            if (tickValue >= min && tickValue <= max) {
                              logTicks.push(tickValue);
                            }
                          }
                        }

                        return logTicks;
                      })()}
                    />
                    <YAxis
                      label={{
                        value: "Value",
                        angle: -90,
                        position: "insideLeft",
                      }}
                      domain={["auto", "auto"]}
                    />
                    <Tooltip
                      content={renderTooltipWithoutRange}
                      isAnimationActive={false}
                    />

                    {/* 3-sigma confidence band */}
                    <Area
                      name="3σ confidence"
                      data={visualization.data.map((d) => ({
                        ...d,
                        a: [d.y_mean - 3 * d.y_std, d.y_mean + 3 * d.y_std],
                      }))}
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
                      data={visualization.data.map((d) => ({
                        ...d,
                        a: [d.y_mean - 2 * d.y_std, d.y_mean + 2 * d.y_std],
                      }))}
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
                      data={visualization.data.map((d) => ({
                        ...d,
                        a: [d.y_mean - 1 * d.y_std, d.y_mean + 1 * d.y_std],
                      }))}
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
                Last computed:{" "}
                {new Date(visualization.computed_at * 1000).toLocaleString()}
              </div>
            </>
          )}
      </CardContent>
    </Card>
  );
};

export default SliceVisualizationCard;
