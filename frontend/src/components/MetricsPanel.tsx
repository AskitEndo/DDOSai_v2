import React from "react";
import { SystemMetrics } from "../types";
import { Activity, Cpu, HardDrive, Wifi } from "lucide-react";
import { formatNumber } from "../utils";

interface MetricsPanelProps {
  metrics: SystemMetrics | null;
  loading?: boolean;
}

const MetricsPanel: React.FC<MetricsPanelProps> = ({
  metrics,
  loading = false,
}) => {
  if (loading) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
          <Activity className="w-5 h-5 mr-2" />
          System Metrics
        </h2>
        <div className="animate-pulse space-y-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-gray-700 h-16 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
        <Activity className="w-5 h-5 mr-2" />
        System Metrics
      </h2>

      <div className="space-y-4">
        <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded">
          <div className="flex items-center">
            <Cpu className="w-4 h-4 text-blue-400 mr-2" />
            <span className="text-gray-300">CPU Usage</span>
          </div>
          <span className="text-white font-medium">
            {metrics?.cpu_usage?.toFixed(1) || 0}%
          </span>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded">
          <div className="flex items-center">
            <HardDrive className="w-4 h-4 text-green-400 mr-2" />
            <span className="text-gray-300">Memory Usage</span>
          </div>
          <span className="text-white font-medium">
            {metrics?.memory_usage?.toFixed(1) || 0}%
          </span>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded">
          <div className="flex items-center">
            <Wifi className="w-4 h-4 text-purple-400 mr-2" />
            <span className="text-gray-300">Active Connections</span>
          </div>
          <span className="text-white font-medium">
            {formatNumber(metrics?.active_connections || 0)}
          </span>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded">
          <div className="flex items-center">
            <Activity className="w-4 h-4 text-yellow-400 mr-2" />
            <span className="text-gray-300">Processing Latency</span>
          </div>
          <span className="text-white font-medium">
            {metrics?.processing_latency_ms || 0}ms
          </span>
        </div>
      </div>
    </div>
  );
};

export default MetricsPanel;
