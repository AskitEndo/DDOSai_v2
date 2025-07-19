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
}

const MetricsPanel: React.FC<MetricsPanelProps> = ({
  metrics,
  loading = false,
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

  // Default values for metrics if they're not available
  const defaultMetrics = {
    cpu_usage: 45,
    memory_usage: 60,
    active_connections: 250,
    processing_latency_ms: 15,
  };

  // Use metrics if available, otherwise use default values
  const cpuUsage = metrics?.cpu_usage ?? defaultMetrics.cpu_usage;
  const memoryUsage = metrics?.memory_usage ?? defaultMetrics.memory_usage;
  const activeConnections =
    metrics?.active_connections ?? defaultMetrics.active_connections;
  const processingLatency =
    metrics?.processing_latency_ms ?? defaultMetrics.processing_latency_ms;
  const maxConnections = 1000; // Assuming a max of 1000 connections for the progress bar

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
        <div className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded-full">
          Real-time
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
