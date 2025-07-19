import React, { useState, useCallback, useEffect } from "react";
import {
  Database,
  RefreshCw,
  Trash2,
  Wifi,
  WifiOff,
  CheckCircle,
  XCircle,
  Cloud,
} from "lucide-react";
import { useDataManager } from "../context/DataManager";
import api from "../services/api";

interface DataControlsProps {
  className?: string;
  showBoth?: boolean;
}

const DataControls: React.FC<DataControlsProps> = ({
  className = "",
  showBoth = true,
}) => {
  const { state, loadDummyData, loadBackendData, clearData } = useDataManager();
  const [isLoading, setIsLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState<
    "checking" | "connected" | "disconnected"
  >("checking");

  // Check backend status on component mount
  useEffect(() => {
    checkBackendStatus();
  }, []);

  const checkBackendStatus = useCallback(async () => {
    setBackendStatus("checking");
    try {
      const response = await api.health();
      if (response.status === 200 && response.data?.status === "healthy") {
        setBackendStatus("connected");
      } else {
        setBackendStatus("disconnected");
      }
    } catch (error) {
      console.error("Backend check failed:", error);
      setBackendStatus("disconnected");
    }
  }, []);

  const handleLoadData = useCallback(async () => {
    setIsLoading(true);
    try {
      await loadDummyData();
      // Short delay for visual feedback
      setTimeout(() => setIsLoading(false), 500);
    } catch (error) {
      console.error("Failed to load dummy data:", error);
      setIsLoading(false);
    }
  }, [loadDummyData]);

  const handleLoadFromBackend = useCallback(async () => {
    setIsLoading(true);
    try {
      await loadBackendData();
      // Short delay for visual feedback
      setTimeout(() => setIsLoading(false), 500);
    } catch (error) {
      console.error("Failed to load backend data:", error);
      setIsLoading(false);
      alert("Failed to load data from backend. Please check your connection.");
    }
  }, [loadBackendData]);

  const handleClearData = useCallback(async () => {
    setIsLoading(true);
    try {
      clearData();
      // Short delay for visual feedback
      setTimeout(() => setIsLoading(false), 300);
    } catch (error) {
      console.error("Failed to clear data:", error);
      setIsLoading(false);
    }
  }, [clearData]);

  const hasData = state.detections.length > 0 || state.metrics !== null;

  const getBackendStatusIcon = () => {
    switch (backendStatus) {
      case "checking":
        return <RefreshCw className="w-3 h-3 mr-1 animate-spin" />;
      case "connected":
        return <CheckCircle className="w-3 h-3 mr-1 text-green-400" />;
      case "disconnected":
        return <XCircle className="w-3 h-3 mr-1 text-red-400" />;
    }
  };

  const getBackendStatusColor = () => {
    switch (backendStatus) {
      case "checking":
        return "bg-gray-900/30 hover:bg-gray-800/50 text-gray-300";
      case "connected":
        return "bg-green-900/30 hover:bg-green-800/50 text-green-300";
      case "disconnected":
        return "bg-red-900/30 hover:bg-red-800/50 text-red-300";
    }
  };

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      {/* Load Data Button */}
      <button
        onClick={handleLoadData}
        disabled={isLoading || (hasData && !showBoth)}
        className="flex items-center px-3 py-1.5 bg-blue-900/30 hover:bg-blue-800/50 text-blue-300 rounded-md text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Load sample data for testing"
      >
        {isLoading ? (
          <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
        ) : (
          <Database className="w-3 h-3 mr-1" />
        )}
        {isLoading ? "Loading..." : "Load Dummy Data"}
      </button>

      {/* Load from Backend Button */}
      <button
        onClick={handleLoadFromBackend}
        disabled={isLoading}
        className="flex items-center px-3 py-1.5 bg-green-900/30 hover:bg-green-800/50 text-green-300 rounded-md text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Load real data from backend"
      >
        {isLoading ? (
          <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
        ) : (
          <Cloud className="w-3 h-3 mr-1" />
        )}
        {isLoading ? "Loading..." : "Load from Backend"}
      </button>

      {/* Clear Data Button */}
      <button
        onClick={handleClearData}
        disabled={isLoading || (!hasData && !showBoth)}
        className="flex items-center px-3 py-1.5 bg-red-900/30 hover:bg-red-800/50 text-red-300 rounded-md text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Clear all sample data"
      >
        {isLoading ? (
          <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
        ) : (
          <Trash2 className="w-3 h-3 mr-1" />
        )}
        {isLoading ? "Clearing..." : "Clear Data"}
      </button>

      {/* Check Backend Button */}
      <button
        onClick={checkBackendStatus}
        disabled={backendStatus === "checking"}
        className={`flex items-center px-3 py-1.5 rounded-md text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${getBackendStatusColor()}`}
        title={`Backend status: ${backendStatus}`}
      >
        {getBackendStatusIcon()}
        {backendStatus === "checking"
          ? "Checking..."
          : backendStatus === "connected"
          ? "Backend OK"
          : "Backend Down"}
      </button>

      {state.usingDummyData && (
        <div className="flex items-center text-xs text-yellow-400 bg-yellow-900/20 px-2 py-1 rounded">
          <div className="w-1.5 h-1.5 bg-yellow-400 rounded-full mr-1.5 animate-pulse"></div>
          Sample Data
        </div>
      )}
    </div>
  );
};

export default DataControls;
