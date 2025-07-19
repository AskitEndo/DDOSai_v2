import React from "react";
import { SystemMetrics } from "../types";
import {
  Activity,
  Cpu,
  HardDrive,
  Wifi,
  BarChart2,
  RefreshCw,
} from "lucide-react";
import { formatNumber } from "../utils";

interface MetricsPanelProps {
  metrics: SystemMetrics | null;
  loading?: boolean;
  onRefresh?: () => void;
}

const MetricsPanel: React.FC<MetricsPanelProps> = ({
  metrics,
  loading = false,
  onRefresh,
}) => {
  const getProgressColor = (value: number, type: string) => {
    if (type === "latency") {
      if (value < 5) return "bg-green-500";
      if (value < 20) return "bg-blue-500";
      if (value < 50) return "bg-yellow-500";
      return "bg-red-500";
    } else {
      if (value < 50) return "bg-green-500";
      if (value < 70) return "bg-blue-500";
      if (value < 85) return "bg-yellow-500";
      return "bg-red-500";
    }
  };

  // Show empty state when no metrics are available
  const showEmpty = !metrics && !loading;

  // Use real metrics when available, show 0 when loading or no data
  const cpuUsage = loading ? 0 : metrics?.cpu_usage ?? 0;
  const memoryUsage = loading ? 0 : metrics?.memory_usage ?? 0;
  const activeConnections = loading ? 0 : metrics?.active_connections ?? 0;
  const processingLatency = loading ? 0 : metrics?.processing_latency_ms ?? 0;
  const maxConnections = 1000;

  if (loading) {
    return (
      <div className="bg-gray-800/70 border border-gray-700 rounded-xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-white flex items-center">
            <BarChart2 className="w-5 h-5 mr-2 text-blue-400" />
            System Metrics
          </h2>
          <RefreshCw className="w-4 h-4 text-gray-400 animate-spin" />
        </div>
        <div className="animate-pulse space-y-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-gray-700/70 h-16 rounded-lg"></div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/70 border border-gray-700 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:border-blue-700/50">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center">
          <BarChart2 className="w-5 h-5 mr-2 text-blue-400" />
          System Metrics
        </h2>
        <div className="flex items-center space-x-2">
          <div className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded-full">
            Real-time
          </div>
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="p-1.5 rounded-full bg-gray-700/50 hover:bg-gray-700 text-gray-400 hover:text-blue-400 transition-colors"
              title="Refresh metrics"
            >
              <RefreshCw className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      <div className="space-y-5">
        {/* CPU Usage */}
        <div className="bg-gray-800/80 p-4 rounded-lg border border-gray-700/50">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center">
              <div className="p-2 bg-blue-900/30 rounded-lg mr-3">
                <Cpu className="w-5 h-5 text-blue-400" />
              </div>
              <span className="text-gray-300 font-medium">CPU Usage</span>
            </div>
            <span className="text-white font-bold text-lg">
              {cpuUsage.toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-700/50 rounded-full h-2.5">
            <div
              className={`h-2.5 rounded-full ${getProgressColor(
                cpuUsage,
                "cpu"
              )}`}
              style={{
                width: `${cpuUsage}%`,
                transition: "width 0.5s ease-in-out",
              }}
            ></div>
          </div>
        </div>

        {/* Memory Usage */}
        <div className="bg-gray-800/80 p-4 rounded-lg border border-gray-700/50">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center">
              <div className="p-2 bg-green-900/30 rounded-lg mr-3">
                <HardDrive className="w-5 h-5 text-green-400" />
              </div>
              <span className="text-gray-300 font-medium">Memory Usage</span>
            </div>
            <span className="text-white font-bold text-lg">
              {memoryUsage.toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-700/50 rounded-full h-2.5">
            <div
              className={`h-2.5 rounded-full ${getProgressColor(
                memoryUsage,
                "memory"
              )}`}
              style={{
                width: `${memoryUsage}%`,
                transition: "width 0.5s ease-in-out",
              }}
            ></div>
          </div>
        </div>

        {/* Active Connections */}
        <div className="bg-gray-800/80 p-4 rounded-lg border border-gray-700/50">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center">
              <div className="p-2 bg-purple-900/30 rounded-lg mr-3">
                <Wifi className="w-5 h-5 text-purple-400" />
              </div>
              <span className="text-gray-300 font-medium">
                Active Connections
              </span>
            </div>
            <span className="text-white font-bold text-lg">
              {formatNumber(activeConnections)}
            </span>
          </div>
          <div className="w-full bg-gray-700/50 rounded-full h-2.5">
            <div
              className="h-2.5 rounded-full bg-purple-500"
              style={{
                width: `${Math.min(
                  (activeConnections / maxConnections) * 100,
                  100
                )}%`,
                transition: "width 0.5s ease-in-out",
              }}
            ></div>
          </div>
        </div>

        {/* Processing Latency */}
        <div className="bg-gray-800/80 p-4 rounded-lg border border-gray-700/50">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center">
              <div className="p-2 bg-yellow-900/30 rounded-lg mr-3">
                <Activity className="w-5 h-5 text-yellow-400" />
              </div>
              <span className="text-gray-300 font-medium">
                Processing Latency
              </span>
            </div>
            <span className="text-white font-bold text-lg">
              {processingLatency}ms
            </span>
          </div>
          <div className="w-full bg-gray-700/50 rounded-full h-2.5">
            <div
              className={`h-2.5 rounded-full ${getProgressColor(
                processingLatency,
                "latency"
              )}`}
              style={{
                width: `${Math.min((processingLatency / 100) * 100, 100)}%`,
                transition: "width 0.5s ease-in-out",
              }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsPanel;
