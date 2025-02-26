import React, { useState, useEffect } from "react";
import { Toaster } from "sonner";
import { Project, Sweep } from "@/lib/types";
import ProjectsView from "./ProjectsView";
import ProjectSweepsView from "./ProjectSweepsView";
import SweepDashboard from "./SweepDashboard";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTheme } from "next-themes";

// View states
type ViewState = 
  | { type: "projects" }
  | { type: "project_sweeps"; project: Project }
  | { type: "sweep_dashboard"; project: Project; sweep: Sweep };

const AppLayout = () => {
  const [viewState, setViewState] = useState<ViewState>({ type: "projects" });
  const [autoRefresh, setAutoRefresh] = useState(false);
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch
  useEffect(() => setMounted(true), []);

  // Navigation handlers
  const navigateToProjects = () => {
    setViewState({ type: "projects" });
  };

  const navigateToProject = (project: Project) => {
    setViewState({ type: "project_sweeps", project });
  };

  const navigateToSweep = (project: Project, sweep: Sweep) => {
    setViewState({ type: "sweep_dashboard", project, sweep });
  };

  // Toggle auto-refresh - this would be passed down to child components
  const toggleAutoRefresh = (value: boolean) => {
    setAutoRefresh(value);
  };

  // Render the appropriate view based on the state
  const renderContent = () => {
    switch (viewState.type) {
      case "projects":
        return (
          <ProjectsView 
            onSelectProject={navigateToProject}
            autoRefresh={autoRefresh}
          />
        );
      case "project_sweeps":
        return (
          <ProjectSweepsView
            project={viewState.project}
            onSelectSweep={(sweep) => navigateToSweep(viewState.project, sweep)}
            onBackToProjects={navigateToProjects}
            autoRefresh={autoRefresh}
          />
        );
      case "sweep_dashboard":
        return (
          <SweepDashboard
            project={viewState.project}
            sweep={viewState.sweep}
            onBackToProject={() => navigateToProject(viewState.project)}
            onBackToProjects={navigateToProjects}
            autoRefresh={autoRefresh}
            onSelectSweep={(sweep) => navigateToSweep(viewState.project, sweep)}
          />
        );
    }
  };

  // Available themes
  const themes = [
    { id: 'light', name: 'Light' },
    { id: 'pastel-light', name: 'Pastel Light' },
    { id: 'sandy-light', name: 'Sandy Light' },
    { id: 'dark', name: 'Dark' },
    { id: 'eco-dark', name: 'Eco-Dark' }
  ];

  return (
    <div className="h-screen max-h-screen overflow-hidden flex flex-col bg-background">
      <header className="bg-card border-b border-border py-4 flex-shrink-0">
        <div className="container mx-auto px-6">
          <div className="flex justify-between items-center">
            <h1 className="text-xl font-bold text-foreground">My Little Optimizer</h1>
            <div className="flex items-center gap-4">
              {/* Theme Selector */}
              {mounted && (
                <div className="flex items-center gap-2">
                  <Select value={theme} onValueChange={setTheme}>
                    <SelectTrigger className="w-[150px]">
                      <SelectValue placeholder="Theme" />
                    </SelectTrigger>
                    <SelectContent>
                      {themes.map(t => (
                        <SelectItem key={t.id} value={t.id}>
                          {t.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
              
              {/* Auto-refresh Toggle */}
              <div className="flex items-center gap-2">
                <Switch
                  id="global-auto-refresh"
                  checked={autoRefresh}
                  onCheckedChange={toggleAutoRefresh}
                />
                <Label htmlFor="global-auto-refresh" className="text-sm text-muted-foreground">
                  Auto-refresh
                </Label>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-hidden">
        {renderContent()}
      </main>

      <Toaster richColors />
    </div>
  );
};

export default AppLayout;