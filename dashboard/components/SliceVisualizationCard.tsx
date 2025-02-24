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
import { Sweep, SliceVisualization, SliceVisualizationResponse, ApiResponse } from "@/lib/types";

interface SliceVisualizationCardProps {
  sweep: Sweep;
}

const SliceVisualizationCard = ({ sweep }: SliceVisualizationCardProps) => {
  const [selectedParam, setSelectedParam] = useState<string | null>(null);
  const [visualization, setVisualization] = useState<SliceVisualization | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedParam) {
      // Set initial parameter selection
      const firstParam = Object.keys(sweep.parameters)[0];
      if (firstParam) {
        setSelectedParam(firstParam);
      }
    }
  }, [sweep.parameters]);

  useEffect(() => {
    const fetchVisualization = async () => {
      if (!selectedParam) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(
          `/api/sweeps/${sweep.id}/visualize/slice`,
          {method: 'POST', body: JSON.stringify({param_name: selectedParam})}
        );
        const result: ApiResponse<SliceVisualizationResponse> = await response.json();
        
        if (result.status === 'ok') {
          let resp = result.data;
          if (resp.cached) {
            setLoading(false);
            setVisualization(resp.cached);
          }
        } else {
          setError(result.message || 'Failed to fetch visualization');
        }
      } catch (err) {
       setError('Failed to fetch visualization');
      }
    };

    fetchVisualization();
  }, [sweep.id, selectedParam]);

  if (!selectedParam) {
    return null;
  }

//   let datapoints = []
//   if (visualization) {
//     datapoints = visualization.data.map((d) => ({...d, x: d.x, y: d.y_mean}))
//   }
  console.log(visualization);

  const renderTooltipWithoutRange = ({ payload, content, ...rest }: TooltipProps<number, string>) => {
        const newPayload = payload.filter((x) => x.dataKey !== "a");
        return <Tooltip payload={newPayload} {...rest} />;
    }

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
        {loading && (
          <div className="h-64 flex items-center justify-center">
            Loading visualization...
          </div>
        )}
        
        {error && (
          <div className="h-64 flex items-center justify-center text-red-500">
            {error}
          </div>
        )}

        {visualization && !loading && !error && (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="x"
                  type="number"
                //   scale="log"
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
                <Tooltip content={renderTooltipWithoutRange} />
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
                //   type="monotone"
                  data={visualization.data}
                  dataKey="y_mean"
                  stroke="#2563eb"
                  dot={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}

        {visualization && (
          <div className="text-sm text-gray-500 mt-2">
            Last computed: {new Date(visualization.computed_at * 1000).toLocaleString()}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SliceVisualizationCard;