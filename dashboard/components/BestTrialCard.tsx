import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Copy, Clock } from "lucide-react";
import { toast } from "sonner";
import { Instant, Duration } from "@js-joda/core";

import type { ExperimentResult, OptimizationObjective } from "@/lib/types";
import assert from "assert";

interface BestTrialCardProps {
  results: ExperimentResult[];
  objective: OptimizationObjective;
}

const BestTrialCard = ({ results, objective = "min" }: BestTrialCardProps) => {
  // Find the best trial based on objective
  const bestTrial = results.reduce(
    (best: ExperimentResult | null, current: ExperimentResult) => {
      if (current.value === null) return best;
      if (current.completed_at === null) return best;
      if (!best) return current;
      return objective === "min"
        ? current.value < best.value
          ? current
          : best
        : current.value > best.value
        ? current
        : best;
    },
    null
  );

  if (bestTrial === null) {
    // Show a greyed out card for no results yet.
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Best Trial</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-gray-500 text-sm">No results yet.</div>
        </CardContent>
      </Card>
    );
  }

  const handleCopyParams = () => {
    navigator.clipboard.writeText(
      JSON.stringify(bestTrial.parameters, null, 2)
    );
    toast.success("Parameters copied to clipboard");
  };

  // Calculate duration
  //   assert(bestTrial.completed_at !== null);
  assert(bestTrial.completed_at !== null);
  const startTime = Instant.ofEpochSecond(bestTrial.created_at);
  const endTime = Instant.ofEpochSecond(bestTrial.completed_at);
  const duration = Duration.between(startTime, endTime);

  // Format duration as hours:minutes:seconds
  const formatDuration = (duration: Duration) => {
    const seconds = duration.seconds();
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return [
      hrs > 0 ? `${hrs}h` : null,
      mins > 0 ? `${mins}m` : null,
      `${secs}s`,
    ]
      .filter(Boolean)
      .join(" ");
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Best Trial</span>
          <Button
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
            onClick={handleCopyParams}
          >
            <Copy className="h-4 w-4" />
            Copy Parameters
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-500">Trial #</p>
            <p className="font-medium">{results.indexOf(bestTrial) + 1}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Value</p>
            <p className="font-medium">{bestTrial.value.toFixed(4)}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Start Time</p>
            <p className="font-medium">{startTime.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Duration</p>
            <p className="font-medium flex items-center gap-2">
              <Clock className="h-4 w-4" />
              {formatDuration(duration)}
            </p>
          </div>
          <div className="col-span-2">
            <p className="text-sm text-gray-500 mb-2">Parameters</p>
            <div className="bg-gray-50 p-3 rounded-md">
              {Object.entries(bestTrial.parameters).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="font-mono text-sm">{key}:</span>
                  <span className="font-mono text-sm">{value.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default BestTrialCard;
