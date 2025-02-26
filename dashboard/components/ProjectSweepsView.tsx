import React, { useState, useEffect } from "react";
import { Calendar, Clock, BarChart2, Activity } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useInterval } from "@/lib/useInterval";
import { Project, Sweep, ApiResponse } from "@/lib/types";
import BreadcrumbNavigator from "./BreadcrumbNavigator";

interface ProjectSweepsViewProps {
  project: Project;
  onSelectSweep: (sweep: Sweep) => void;
  onBackToProjects: () => void;
  autoRefresh: boolean;
}

const ProjectSweepsView = ({ 
  project, 
  onSelectSweep, 
  onBackToProjects, 
  autoRefresh 
}: ProjectSweepsViewProps) => {
  const [sweeps, setSweeps] = useState<Sweep[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchSweeps = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/projects/${project.id}/sweeps`);
      const data: ApiResponse<Sweep[]> = await response.json();
      if (data.status !== "ok") {
        throw new Error(data.message);
      }
      setSweeps(data.data.sort((a, b) => b.created_at - a.created_at));
    } catch (error) {
      console.error("Error fetching sweeps:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSweeps();
  }, [project.id]);

  // Auto refresh
  const REFRESH_INTERVAL = 5000; // 5 seconds
  useInterval(
    () => {
      fetchSweeps();
    },
    autoRefresh ? REFRESH_INTERVAL : null
  );

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="bg-card border-b border-border p-4 flex justify-between items-center">
        <BreadcrumbNavigator
          items={[
            {
              label: project.name,
              onClick: () => {}, // Already on this view
            },
          ]}
          onHomeClick={onBackToProjects}
        />
      </div>
      
      <div className="container mx-auto p-6 overflow-auto bg-background">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-foreground">{project.name}</h1>
          <div className="text-sm text-muted-foreground">
            Created {new Date(project.created_at * 1000).toLocaleString()}
          </div>
        </div>

      <Card>
        <CardHeader>
          <CardTitle>Sweeps</CardTitle>
        </CardHeader>
        <CardContent>
          {loading && sweeps.length === 0 ? (
            <div className="text-center p-4 text-foreground">Loading sweeps...</div>
          ) : sweeps.length === 0 ? (
            <div className="text-center text-muted-foreground p-4">
              No sweeps in this project yet.
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Trials</TableHead>
                  <TableHead>Objective</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sweeps.map((sweep) => (
                  <TableRow
                    key={sweep.id}
                    className="cursor-pointer hover:bg-accent"
                    onClick={() => onSelectSweep(sweep)}
                  >
                    <TableCell className="font-medium">{sweep.name}</TableCell>
                    <TableCell>
                      <div className="flex items-center">
                        <BarChart2 className="h-4 w-4 mr-2 text-primary" />
                        {sweep.num_trials}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center">
                        <Activity className="h-4 w-4 mr-2 text-primary" />
                        {sweep.objective === "min" ? "Minimize" : "Maximize"}
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className="px-2 py-1 rounded-full text-xs bg-primary/20 text-primary">
                        {sweep.status}
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center text-muted-foreground">
                        <Calendar className="h-4 w-4 mr-2" />
                        {new Date(sweep.created_at * 1000).toLocaleDateString()}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
      </div>
    </div>
  );
};

export default ProjectSweepsView;