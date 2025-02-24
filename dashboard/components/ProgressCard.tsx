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
}

const ProgressCard = ({ results, objective }: ProgressCardProps) => {
  if (!results || results.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Optimization Progress</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-gray-500 text-sm">No results yet.</div>
        </CardContent>
      </Card>
    );
  }

  let bestTrials: TrialResult[] = [];
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
    bestTrials.push({ ...result, value: best.value });
  }

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Optimization Progress</CardTitle>
      </CardHeader>
      <CardContent>
        {bestTrials.length} trials so far.
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
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
              />
              <Tooltip animationDuration={0} />
              <Line
                name="Best so far"
                type="stepAfter"
                data={bestTrials}
                dataKey="value"
                key="best"
                stroke="#2563eb"
                dot={false}
                isAnimationActive={false}
              />
              <Scatter
                name="Trial results"
                data={results}
                dataKey="value"
                key="results"
                fill="#6366f1"
                isAnimationActive={false}
                stroke="#8884d8"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default ProgressCard;
