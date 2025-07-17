import React from "react";
import { SimulationStatus } from "../types";
import { formatTimestamp, formatNumber, formatBytes } from "../utils";
import { History, CheckCircle, AlertTriangle, Clock } from "lucide-react";

interface SimulationHistoryPanelProps {
  history: SimulationStatus[];
  loading?: boolean;
  onSelectSimulation?: (simulation: SimulationStatus) => void;
}

const SimulationHistoryPanel: React.FC<SimulationHistoryPanelProps> = ({
  history,
  loading = false,
  onSelectSimulation,
}) => {
  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="w-4 h-4 text-success" />;
      case "error":
        return <AlertTriangle className="w-4 h-4 text-danger" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "text-success";
      case "error":
        return "text-danger";
      default:
        return "text-gray-400";
    }
  };

  if (loading) {
    return (
      <div className="bg-dark-surface border border-dark-border rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
          <History className="w-5 h-5 mr-2" />
          Simulation History
        </h2>
        <div className="animate-pulse space-y-4">
          <div className="h-12 bg-gray-700 rounded"></div>
          <div className="h-12 bg-gray-700 rounded"></div>
          <div className="h-12 bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-dark-surface border border-dark-border rounded-lg p-6">
      <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
        <History className="w-5 h-5 mr-2" />
        Simulation History
      </h2>

      {history.length > 0 ? (
        <div className="space-y-3">
          {history.map((simulation) => (
            <div
              key={simulation.simulation_id}
              className="bg-gray-800/50 p-3 rounded-md hover:bg-gray-800 transition-colors cursor-pointer"
              onClick={() => onSelectSimulation?.(simulation)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {getStatusIcon(simulation.status)}
                  <span
                    className={`font-medium ${getStatusColor(
                      simulation.status
                    )}`}
                  >
                    {simulation.attack_type?.replace("_", " ").toUpperCase() ||
                      "Unknown"}
                  </span>
                </div>
                <div className="text-sm text-gray-400">
                  {simulation.start_time
                    ? formatTimestamp(simulation.start_time)
                    : "N/A"}
                </div>
              </div>

              <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                <div>
                  <span className="text-gray-400">Target: </span>
                  <span className="text-white">
                    {simulation.target_ip}:{simulation.target_port}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Packets: </span>
                  <span className="text-white">
                    {formatNumber(simulation.packets_sent)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Data: </span>
                  <span className="text-white">
                    {formatBytes(simulation.bytes_sent)}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-6 text-gray-400">
          <p>No simulation history available</p>
        </div>
      )}
    </div>
  );
};

export default SimulationHistoryPanel;
