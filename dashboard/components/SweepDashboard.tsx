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
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useInterval } from "@/lib/useInterval";
import { Toaster } from "sonner";

import BestTrialCard from "./BestTrialCard";
import ProgressCard from "./ProgressCard";
import TrialListCard from "./TrialListCard";
import { TrialResult, Sweep, ApiResponse } from "@/lib/types";
import SliceVisualizationCard from "./SliceVisualizationCard";

const SweepDashboard = () => {
  const [sweeps, setSweeps] = useState<Sweep[]>([]);
  const [selectedSweep, setSelectedSweep] = useState<Sweep | null>(null);
  const [results, setResults] = useState<TrialResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [sweepToRename, setSweepToRename] = useState<Sweep | null>(null);
  const [newName, setNewName] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(false);

  useEffect(() => {
    fetchSweeps();
  }, []);

  const fetchSweeps = async () => {
    try {
      const response = await fetch("/api/sweeps");
      const data: ApiResponse<Sweep[]> = await response.json();
      if (data.status !== 'ok') {
        throw new Error(data.message);
      }
      setSweeps(data.data);
    } catch (error) {
      console.error("Error fetching sweeps:", error);
    }
  };

  const fetchResults = async () => {
    if (!selectedSweep) return;

    try {
      setLoading(true);
      const response = await fetch(`/api/sweeps/${selectedSweep.id}/trials`);
      const data: ApiResponse<TrialResult[]> = await response.json();
      if (data.status !== 'ok') {
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

  useEffect(() => {
    fetchResults();
  }, [selectedSweep]);

  // Use the interval hook to periodically fetch data
  const REFRESH_INTERVAL = 5000;
  useInterval(
    () => {
      fetchSweeps();
      if (selectedSweep) {
        fetchResults();
      }
    },
    autoRefresh ? REFRESH_INTERVAL : null
  );

  const handleDelete = async (
    sweep: Sweep,
    e: React.MouseEvent<HTMLElement>
  ) => {
    e.stopPropagation(); // Prevent sweep selection when clicking menu

    if (!confirm(`Are you sure you want to delete sweep "${sweep.name}"?`)) {
      return;
    }

    try {
      const response = await fetch(`/api/sweeps/${sweep.id}`, {
        method: "DELETE",
      });

      if (!response.ok) throw new Error("Failed to delete sweep");

      // Refresh sweeps list
      await fetchSweeps();

      // Clear selected sweep if it was the one we just deleted
      if (selectedSweep?.id === sweep.id) {
        setSelectedSweep(null);
      }
    } catch (error) {
      console.error("Error deleting sweep:", error);
    }
  };

  const handleRename = async () => {
    if (!sweepToRename || !newName.trim()) return;

    try {
      const response = await fetch(`/api/sweeps/${sweepToRename.id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: newName.trim(),
        }),
      });

      if (!response.ok) throw new Error("Failed to rename sweep");

      // Refresh sweeps list
      await fetchSweeps();

      // Update selected sweep if it was the one renamed
      if (selectedSweep?.id === sweepToRename.id) {
        setSelectedSweep({ ...selectedSweep, name: newName.trim() });
      }

      setRenameDialogOpen(false);
      setSweepToRename(null);
      setNewName("");
    } catch (error) {
      console.error("Error renaming sweep:", error);
    }
  };

  const openRenameDialog = (sweep: Sweep, e: React.MouseEvent<HTMLElement>) => {
    e.stopPropagation(); // Prevent sweep selection when clicking menu
    setSweepToRename(sweep);
    setNewName(sweep.name);
    setRenameDialogOpen(true);
  };

  return (
    <div className="flex h-screen">
      <Toaster richColors />
      {/* Sidebar */}
      <div className="w-64 bg-gray-100 p-4 border-r">
        <h2 className="text-xl font-bold mb-4">Sweeps</h2>
        <div className="flex items-center gap-2">
          <Switch
            checked={autoRefresh}
            onCheckedChange={setAutoRefresh}
            id="auto-refresh"
          />
          <Label htmlFor="auto-refresh" className="text-sm">
            Auto-refresh
          </Label>
        </div>
        <div className="space-y-2">
          {sweeps.map((sweep) => (
            <div
              key={sweep.id}
              className={`flex items-center justify-between p-2 rounded cursor-pointer ${
                selectedSweep?.id === sweep.id
                  ? "bg-blue-500 text-white"
                  : "hover:bg-gray-200"
              }`}
              onClick={() => setSelectedSweep(sweep)}
            >
              <span className="truncate mr-2">{sweep.name}</span>
              <DropdownMenu>
                <DropdownMenuTrigger
                  asChild
                  onClick={(e) => e.stopPropagation()}
                >
                  <Button
                    variant="ghost"
                    size="icon"
                    className={
                      selectedSweep?.id === sweep.id
                        ? "hover:bg-blue-600"
                        : "hover:bg-gray-300"
                    }
                  >
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem onClick={(e) => openRenameDialog(sweep, e)}>
                    Rename
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    className="text-red-600"
                    onClick={(e) => handleDelete(sweep, e)}
                  >
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          ))}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 p-4 overflow-auto">
        {selectedSweep ? (
          <>
            <h1 className="text-2xl font-bold mb-4">
              Sweep: {selectedSweep.name}
            </h1>

            <div className="flex gap-6 mb-6">
              <div className="w-1/3">
                <BestTrialCard
                  results={results}
                  objective={selectedSweep.objective}
                />
              </div>

              {/* Results Chart */}
              <div className="w-2/3">
                <ProgressCard
                  results={results}
                  objective={selectedSweep.objective}
                />
              </div>
            </div>

            <SliceVisualizationCard sweep={selectedSweep} autoRefresh={autoRefresh} />

            {/* Results Table */}
            <TrialListCard results={results} />
          </>
        ) : (
          <div className="text-center text-gray-500 mt-10">
            Select a sweep from the sidebar to view results
          </div>
        )}
      </div>

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
            <Button
              variant="outline"
              onClick={() => setRenameDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleRename}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default SweepDashboard;
