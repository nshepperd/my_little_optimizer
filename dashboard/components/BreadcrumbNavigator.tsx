import React from "react";
import { ChevronRight, Home } from "lucide-react";
import { Button } from "@/components/ui/button";

interface BreadcrumbItem {
  label: string;
  onClick: () => void;
}

interface BreadcrumbNavigatorProps {
  items: BreadcrumbItem[];
  onHomeClick: () => void;
  className?: string;
}

const BreadcrumbNavigator = ({ items, onHomeClick, className = "" }: BreadcrumbNavigatorProps) => {
  return (
    <div className={`flex items-center space-x-1 ${className}`}>
      <Button
        variant="ghost" 
        size="sm"
        className="flex items-center text-muted-foreground hover:text-foreground"
        onClick={onHomeClick}
      >
        <Home className="h-4 w-4 mr-1" />
        Projects
      </Button>
      
      {items.map((item, index) => (
        <React.Fragment key={index}>
          <ChevronRight className="h-4 w-4 text-muted-foreground" />
          <Button
            variant="ghost"
            size="sm"
            className="flex items-center text-muted-foreground hover:text-foreground"
            onClick={item.onClick}
          >
            {item.label}
          </Button>
        </React.Fragment>
      ))}
    </div>
  );
};

export default BreadcrumbNavigator;