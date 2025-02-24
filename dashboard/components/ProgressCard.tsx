import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ExperimentResult } from "@/lib/types";

interface ProgressCardProps {
  results: ExperimentResult[];
}

const ProgressCard = ({ results }: ProgressCardProps) => {
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

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Optimization Progress</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={results}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="index"
                label={{ value: "Experiment Number", position: "bottom" }}
              />
              <YAxis
                label={{
                  value: "Value",
                  angle: -90,
                  position: "insideLeft",
                }}
              />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#8884d8"
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default ProgressCard;
