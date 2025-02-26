import React, { useState, useEffect } from "react";
import { Calendar, BarChart2 } from "lucide-react";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useInterval } from "@/lib/useInterval";
import { Project, ApiResponse } from "@/lib/types";
import BreadcrumbNavigator from "./BreadcrumbNavigator";

interface ProjectsViewProps {
  onSelectProject: (project: Project) => void;
  autoRefresh: boolean;
}

const ProjectsView = ({ onSelectProject, autoRefresh }: ProjectsViewProps) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchProjects = async () => {
    try {
      setLoading(true);
      const response = await fetch("/api/projects");
      const data: ApiResponse<Project[]> = await response.json();
      if (data.status !== "ok") {
        throw new Error(data.message);
      }
      setProjects(data.data.sort((a, b) => b.created_at - a.created_at));
    } catch (error) {
      console.error("Error fetching projects:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProjects();
  }, []);

  // Auto refresh
  const REFRESH_INTERVAL = 10000; // 10 seconds
  useInterval(
    () => {
      fetchProjects();
    },
    autoRefresh ? REFRESH_INTERVAL : null
  );

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="bg-card border-b border-border p-4 flex justify-between items-center">
        <BreadcrumbNavigator items={[]} onHomeClick={() => {}} />
      </div>
      
      <div className="container mx-auto p-6 overflow-auto bg-background">
        <h1 className="text-3xl font-bold mb-6 text-foreground">Projects</h1>

      <Card>
        <CardHeader>
          <CardTitle>All Projects</CardTitle>
        </CardHeader>
        <CardContent>
          {loading && projects.length === 0 ? (
            <div className="text-center p-4 text-foreground">Loading projects...</div>
          ) : projects.length === 0 ? (
            <div className="text-center text-muted-foreground p-4">
              No projects yet. Create a project by starting a sweep via the Python client.
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Sweeps</TableHead>
                  <TableHead>Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {projects.map((project) => (
                  <TableRow
                    key={project.id}
                    className="cursor-pointer hover:bg-accent"
                    onClick={() => onSelectProject(project)}
                  >
                    <TableCell className="font-medium">{project.name}</TableCell>
                    <TableCell>
                      <div className="flex items-center">
                        <BarChart2 className="h-4 w-4 mr-2 text-primary" />
                        {project.sweep_count}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center text-muted-foreground">
                        <Calendar className="h-4 w-4 mr-2" />
                        {new Date(project.created_at * 1000).toLocaleDateString()}
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

export default ProjectsView;