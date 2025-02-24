import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrialResult } from "@/lib/types";

interface TrialListCardProps {
  results: TrialResult[];
}

const TrialListCard = ({ results }: TrialListCardProps) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Trials</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="p-2 text-left">#</th>
                <th className="p-2 text-left">Parameters</th>
                <th className="p-2 text-left">Value</th>
                <th className="p-2 text-left">Status</th>
                <th className="p-2 text-left">Created</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result, index) => (
                <tr key={index} className="border-b">
                  <td className="p-2">{index + 1}</td>
                  <td className="p-2">
                    {Object.entries(result.parameters)
                      .map(([key, value]) => `${key}: ${value.toFixed(4)}`)
                      .join(", ")}
                  </td>
                  <td className="p-2">
                    {result.value !== null ? result.value.toFixed(4) : "N/A"}
                  </td>
                  <td className="p-2">{result.status}</td>
                  <td className="p-2">
                    {new Date(result.created_at * 1000).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
};

export default TrialListCard;
