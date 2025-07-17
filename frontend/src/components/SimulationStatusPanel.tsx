import React from "react";
import { SimulationStatus } from "../types";
import { formatNumber, formatBytes } from "../utils";
import { AlertTriangle, CheckCircle, Clock, Zap } from "lucide-react";
import LoadingSpinner from "./LoadingSpinner";

interface SimulationStatusPanelProps {
  simulation: SimulationStatus | null;
  loading?: boolean;
  onRefresh?: () => void;
}

const SimulationStatusPanel: React.FC<SimulationStatusPanelProps> = ({
  simulation,
  loading = false,
  onRefresh,
}) => {
  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case "running":
        return "text-warning";
      case "completed":
        return "text-success";
      case "error":
        return "text-danger";
      default:
        return "text-gray-400";
    }
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running":
        return <Zap className="w-5 h-5 text-warning" />;
      case "completed":
        return <CheckCircle className="w-5 h-5 text-success" />;
      case "error":
        return <AlertTriangle className="w-5 h-5 text-danger" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  if (loading) {
    return (
      <div className="bg-dark-surface border border-dark-border rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">
          Simulation Status
        </h2>
        <div className="flex items-center justify-center py-8">
          <LoadingSpinner size="medium" text="Loading status..." />
        </div>
      </div>
    );
  }

  if (!simulation) {
    return (
      <div className="bg-dark-surface border border-dark-border rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">
          Simulation Status
        </h2>
        <div className="text-center py-8 text-gray-400">
          <p>No active simulation</p>
          <p className="text-sm mt-2">
            Configure and start a simulation to see status here
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-dark-surface border border-dark-border rounded-lg p-6">
      <h2 className="text-xl font-semibold text-white mb-4">
        Simulation Status
      </h2>

      <div className="space-y-6">
        {/* Status Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {getStatusIcon(simulation.status)}
            <span
              className={`font-medium ${getStatusColor(simulation.status)}`}
            >
              {simulation.status.charAt(0).toUpperCase() +
                simulation.status.slice(1)}
            </span>
          </div>
          <div className="text-sm text-gray-400">
            ID: {simulation.simulation_id || "N/A"}
          </div>
        </div>

        {/* Progress Bar */}
        {simulation.status === "running" &&
          simulation.elapsed_time !== undefined && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Progress</span>
                <span className="text-white">
                  {Math.min(
                    Math.round(
                      (simulation.elapsed_time / simulation.duration) * 100
                    ),
                    100
                  )}
                  %
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.min(
                      Math.round(
                        (simulation.elapsed_time / simulation.duration) * 100
                      ),
                      100
                    )}%`,
                  }}
                />
              </div>
            </div>
          )}

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <div className="text-sm text-gray-400">Attack Type</div>
            <div className="text-white font-medium mt-1">
              {simulation.attack_type
                ? simulation.attack_type.replace("_", " ").toUpperCase()
                : "N/A"}
            </div>
          </div>
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <div className="text-sm text-gray-400">Target</div>
            <div className="text-white font-medium mt-1">
              {simulation.target_ip}:{simulation.target_port}
            </div>
          </div>
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <div className="text-sm text-gray-400">Packets Sent</div>
            <div className="text-white font-medium mt-1">
              {formatNumber(simulation.packets_sent)}
            </div>
          </div>
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <div className="text-sm text-gray-400">Data Sent</div>
            <div className="text-white font-medium mt-1">
              {formatBytes(simulation.bytes_sent)}
            </div>
          </div>
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <div className="text-sm text-gray-400">Duration</div>
            <div className="text-white font-medium mt-1">
              {simulation.duration} seconds
            </div>
          </div>
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <div className="text-sm text-gray-400">Packet Rate</div>
            <div className="text-white font-medium mt-1">
              {formatNumber(
                simulation.current_packet_rate || simulation.packet_rate
              )}{" "}
              pps
            </div>
          </div>
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <div className="text-sm text-gray-400">Start Time</div>
            <div className="text-white font-medium mt-1">
              {simulation.start_time
                ? new Date(simulation.start_time).toLocaleTimeString()
                : "N/A"}
            </div>
          </div>
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <div className="text-sm text-gray-400">End Time</div>
            <div className="text-white font-medium mt-1">
              {simulation.end_time
                ? new Date(simulation.end_time).toLocaleTimeString()
                : "N/A"}
            </div>
          </div>
        </div>

        {/* Errors */}
        {simulation.errors > 0 && (
          <div className="bg-danger/10 border border-danger/30 rounded-lg p-4 text-danger flex items-start">
            <AlertTriangle className="w-5 h-5 mr-3 mt-0.5 flex-shrink-0" />
            <div>
              <p className="font-medium">Simulation Errors Detected</p>
              <p className="text-sm mt-1">
                {simulation.errors} errors occurred during simulation execution.
                Check logs for more details.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SimulationStatusPanel;
