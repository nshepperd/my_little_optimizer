import React, { useState, useEffect } from "react";
import { MoreVertical } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { useInterval } from "@/lib/useInterval";

import BestTrialCard from "./BestTrialCard";
import ProgressCard from "./ProgressCard";
import TrialListCard from "./TrialListCard";
import BreadcrumbNavigator from "./BreadcrumbNavigator";
import { TrialResult, Sweep, Project, ApiResponse } from "@/lib/types";
import SliceVisualizationCard from "./SliceVisualizationCard";

interface SweepDashboardProps {
  sweep: Sweep;
  project: Project;
  onBackToProject: () => void;
  onBackToProjects: () => void;
  autoRefresh: boolean;
  onSelectSweep: (sweep: Sweep) => void;
}

const SweepDashboard = ({
  sweep,
  project,
  onBackToProject,
  onBackToProjects,
  autoRefresh,
  onSelectSweep,
}: SweepDashboardProps) => {
  const [results, setResults] = useState<TrialResult[]>([]);
  const [sweeps, setSweeps] = useState<Sweep[]>([]);
  const [loading, setLoading] = useState(true);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [newName, setNewName] = useState("");

  const fetchResults = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/sweeps/${sweep.id}/trials/`);
      const data: ApiResponse<TrialResult[]> = await response.json();
      if (data.status !== "ok") {
        throw new Error(data.message);
      }
      const sortedData = data.data
        .sort((a, b) => a.created_at - b.created_at)
        .map((result, index) => ({
          ...result,
          index: index + 1,
        }));
      setResults(sortedData);
    } catch (error) {
      console.error("Error fetching results:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchSweeps = async () => {
    try {
      const response = await fetch(`/api/projects/${project.id}/sweeps`);
      const data: ApiResponse<Sweep[]> = await response.json();
      if (data.status !== "ok") {
        throw new Error(data.message);
      }
      setSweeps(data.data.sort((a, b) => b.created_at - a.created_at));
    } catch (error) {
      console.error("Error fetching sweeps:", error);
    }
  };

  useEffect(() => {
    fetchResults();
    fetchSweeps();
    setNewName(sweep.name);
  }, [sweep.id, project.id]);

  // Use the interval hook to periodically fetch data
  const REFRESH_INTERVAL = 5000;
  useInterval(
    () => {
      fetchResults();
      fetchSweeps();
    },
    autoRefresh ? REFRESH_INTERVAL : null
  );

  const handleRename = async () => {
    if (!newName.trim()) return;

    try {
      const response = await fetch(`/api/sweeps/${sweep.id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: newName.trim(),
        }),
      });

      if (!response.ok) throw new Error("Failed to rename sweep");

      setRenameDialogOpen(false);
    } catch (error) {
      console.error("Error renaming sweep:", error);
    }
  };

  return (
    <div className="flex h-screen max-h-screen overflow-hidden">
      {/* Sidebar */}
      <div className="w-64 bg-muted border-r border-border overflow-y-auto">
        <div className="p-4">
          <h2 className="text-xl font-bold mb-4 text-foreground">Sweeps</h2>
          <div className="space-y-2">
            {sweeps.map((s) => (
              <div
                key={s.id}
                className={`flex items-center justify-between p-2 rounded cursor-pointer ${
                  sweep.id === s.id
                    ? "bg-primary text-primary-foreground"
                    : "hover:bg-accent hover:text-accent-foreground"
                }`}
                onClick={() => onSelectSweep(s)}
              >
                <span className="truncate mr-2">{s.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="bg-card border-b border-border p-4 flex justify-between items-center">
          <BreadcrumbNavigator
            items={[
              {
                label: project.name,
                onClick: onBackToProject,
              },
              {
                label: sweep.name,
                onClick: () => {}, // Already on this view
              },
            ]}
            onHomeClick={onBackToProjects}
          />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                Actions <MoreVertical className="h-4 w-4 ml-1" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem onClick={() => setRenameDialogOpen(true)}>
                Rename
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        
        <div className="p-6 overflow-auto">
          <div className="mb-6">
            <h1 className="text-2xl font-bold text-foreground">{sweep.name}</h1>
          </div>

        <div className="flex gap-6 mb-6">
          <BestTrialCard
            results={results}
            objective={sweep.objective}
            className="flex-0"
          />

          {/* Results Chart */}
          <ProgressCard
            results={results}
            objective={sweep.objective}
            className="flex-1"
          />
        </div>

        <SliceVisualizationCard sweep={sweep} autoRefresh={autoRefresh} />

        {/* Results Table */}
        <TrialListCard results={results} />

        {/* Rename Dialog */}
        <Dialog open={renameDialogOpen} onOpenChange={setRenameDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Rename Sweep</DialogTitle>
            </DialogHeader>
            <Input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Enter new name"
            />
            <DialogFooter>
              <Button variant="outline" onClick={() => setRenameDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleRename}>Save</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
    </div>
  );
};

export default SweepDashboard;