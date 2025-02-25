import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Scatter,
  ComposedChart,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrialResult, OptimizationObjective } from "@/lib/types";

interface ProgressCardProps {
  results: TrialResult[];
  objective: OptimizationObjective;
  className?: string;
}

const ProgressCard = ({ results, objective, className }: ProgressCardProps) => {
  if (!results || results.length === 0) {
    return (
      <Card className={`${className || ""}`}>
        <CardHeader>
          <CardTitle>Optimization Progress</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-gray-500 text-sm">No results yet.</div>
        </CardContent>
      </Card>
    );
  }

  let bestTrials = [];
  let best: TrialResult | null = null;
  for (const result of results) {
    if (result.value === null) continue;
    if (
      best === null ||
      (objective === "min" && result.value < best.value) ||
      (objective === "max" && result.value > best.value)
    ) {
      best = result;
    }
    bestTrials.push({ ...result, best_value: best.value });
  }

  return (
    <Card className={`${className || ""} flex flex-col`}>
      <CardHeader>
        <CardTitle>Optimization Progress</CardTitle>
      </CardHeader>
      <CardContent className="gap-4 flex-1">
        <div className="mb-2">{bestTrials.length} trials so far.</div>
        <ResponsiveContainer minWidth={200} width="100%" height="80%">
          <ComposedChart
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            data={bestTrials}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="trial_number"
              type="number"
              label={{ value: "Trial Number", position: "bottom" }}
            />
            <YAxis
              dataKey="value"
              label={{
                value: "Value",
                angle: -90,
                position: "insideLeft",
              }}
              domain={["auto", "auto"]}
            />
            <Tooltip animationDuration={0} />
            <Line
              name="Best so far"
              type="stepAfter"
              dataKey="best_value"
              key="best"
              stroke="#2563eb"
              dot={false}
              isAnimationActive={false}
            />
            <Scatter
              name="Trial results"
              dataKey="value"
              key="results"
              fill="#6366f1"
              isAnimationActive={false}
              stroke="#8884d8"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default ProgressCard;
