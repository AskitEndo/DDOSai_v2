import React, { useState, useEffect } from "react";
import { useApi } from "../hooks/useApi";
import api from "../services/api";
import { Activity, Cpu, Clock, AlertTriangle } from "lucide-react";
import { formatNumber } from "../utils";

interface PerformanceMonitorProps {
  refreshInterval?: number;
}

const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  refreshInterval = 5000,
}) => {
  const [showDetailed, setShowDetailed] = useState(false);

  const {
    data: metrics,
    loading,
    refresh,
  } = useApi(() => api.getSystemMetrics(showDetailed), {
    immediate: true,
    refreshInterval,
  });

  const detailedMetrics = metrics?.data;

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center">
          <Activity className="w-5 h-5 mr-2" />
          Performance Monitoring
        </h2>
        <div className="flex items-center">
          <button
            onClick={() => setShowDetailed(!showDetailed)}
            className="px-3 py-1 text-sm bg-gray-700 hover:bg-gray-600 rounded-md mr-2 transition-colors"
          >
            {showDetailed ? "Basic View" : "Detailed View"}
          </button>
          <button
            onClick={refresh}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            disabled={loading}
          >
            <svg
              className={`w-4 h-4 ${loading ? "animate-spin" : ""}`}
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
          </button>
        </div>
      </div>

      {loading ? (
        <div className="animate-pulse space-y-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-gray-700 h-16 rounded"></div>
          ))}
        </div>
      ) : !metrics?.data ? (
        <div className="text-center text-gray-400 py-8">
          <AlertTriangle className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>Failed to load metrics data</p>
        </div>
      ) : showDetailed ? (
        <DetailedMetricsView metrics={detailedMetrics} />
      ) : (
        <BasicMetricsView metrics={detailedMetrics} />
      )}
    </div>
  );
};

const BasicMetricsView: React.FC<{ metrics: any }> = ({ metrics }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="bg-gray-700/50 rounded p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Cpu className="w-4 h-4 text-blue-400 mr-2" />
            <span className="text-gray-300">System Resources</span>
          </div>
          <div className="text-right">
            <div className="text-white font-medium">
              CPU: {metrics.cpu_usage?.toFixed(1) || 0}%
            </div>
            <div className="text-white font-medium">
              Memory: {metrics.memory_usage?.toFixed(1) || 0}%
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-700/50 rounded p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Clock className="w-4 h-4 text-green-400 mr-2" />
            <span className="text-gray-300">Processing</span>
          </div>
          <div className="text-right">
            <div className="text-white font-medium">
              Latency: {metrics.processing_latency_ms || 0} ms
            </div>
            <div className="text-white font-medium">
              Packets: {formatNumber(metrics.packets_processed || 0)}
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-700/50 rounded p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <AlertTriangle className="w-4 h-4 text-yellow-400 mr-2" />
            <span className="text-gray-300">Threat Level</span>
          </div>
          <div className="text-right">
            <div className="text-white font-medium">
              Level: {metrics.threat_level || 0}/5
            </div>
            <div className="text-white font-medium">
              Malicious: {formatNumber(metrics.malicious_packets || 0)}
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-700/50 rounded p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Activity className="w-4 h-4 text-purple-400 mr-2" />
            <span className="text-gray-300">Connections</span>
          </div>
          <div className="text-right">
            <div className="text-white font-medium">
              Active: {metrics.active_connections || 0}
            </div>
            <div className="text-white font-medium">
              Total: {formatNumber(metrics.total_detections || 0)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const DetailedMetricsView: React.FC<{ metrics: any }> = ({ metrics }) => {
  if (!metrics || !metrics.packets) {
    return (
      <div className="text-center text-gray-400 py-8">
        <p>No detailed metrics available</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* System Metrics */}
      <div>
        <h3 className="text-lg font-medium text-white mb-3">System Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-700/50 rounded p-3">
            <p className="text-gray-400 text-sm">CPU Usage</p>
            <p className="text-white text-xl font-medium">
              {metrics.system?.cpu_usage?.toFixed(1) || 0}%
            </p>
          </div>
          <div className="bg-gray-700/50 rounded p-3">
            <p className="text-gray-400 text-sm">Memory Usage</p>
            <p className="text-white text-xl font-medium">
              {metrics.system?.memory_usage?.toFixed(1) || 0}%
            </p>
          </div>
          <div className="bg-gray-700/50 rounded p-3">
            <p className="text-gray-400 text-sm">Disk Usage</p>
            <p className="text-white text-xl font-medium">
              {metrics.system?.disk_usage?.toFixed(1) || 0}%
            </p>
          </div>
        </div>
      </div>

      {/* Packet Processing */}
      <div>
        <h3 className="text-lg font-medium text-white mb-3">
          Packet Processing
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-700/50 rounded p-3">
            <p className="text-gray-400 text-sm">Total Packets</p>
            <p className="text-white text-xl font-medium">
              {formatNumber(metrics.packets?.total || 0)}
            </p>
          </div>
          <div className="bg-gray-700/50 rounded p-3">
            <p className="text-gray-400 text-sm">Malicious Packets</p>
            <p className="text-white text-xl font-medium">
              {formatNumber(metrics.packets?.malicious || 0)}
            </p>
            <p className="text-gray-400 text-xs">
              {(
                (metrics.packets?.malicious / metrics.packets?.total) * 100 || 0
              ).toFixed(2)}
              % of total
            </p>
          </div>
          <div className="bg-gray-700/50 rounded p-3">
            <p className="text-gray-400 text-sm">Processing Time</p>
            <p className="text-white text-xl font-medium">
              {(metrics.processing?.avg_time_ms || 0).toFixed(2)} ms
            </p>
            <p className="text-gray-400 text-xs">
              Throughput: {(metrics.processing?.throughput || 0).toFixed(2)}{" "}
              pkt/s
            </p>
          </div>
        </div>
      </div>

      {/* Model Performance */}
      {metrics.models && Object.keys(metrics.models).length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-white mb-3">
            Model Performance
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(metrics.models).map(
              ([modelName, modelData]: [string, any]) => (
                <div key={modelName} className="bg-gray-700/50 rounded p-3">
                  <p className="text-gray-400 text-sm">{modelName}</p>
                  <p className="text-white text-xl font-medium">
                    {(modelData.avg_time * 1000).toFixed(2)} ms
                  </p>
                  <div className="flex justify-between text-xs text-gray-400 mt-1">
                    <span>
                      Min: {(modelData.min_time * 1000).toFixed(2)} ms
                    </span>
                    <span>
                      Max: {(modelData.max_time * 1000).toFixed(2)} ms
                    </span>
                    <span>Count: {modelData.count}</span>
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      )}

      {/* Error Metrics */}
      <div>
        <h3 className="text-lg font-medium text-white mb-3">Error Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-700/50 rounded p-3">
            <p className="text-gray-400 text-sm">Error Count</p>
            <p className="text-white text-xl font-medium">
              {formatNumber(metrics.errors?.count || 0)}
            </p>
          </div>
          <div className="bg-gray-700/50 rounded p-3">
            <p className="text-gray-400 text-sm">Error Rate</p>
            <p className="text-white text-xl font-medium">
              {(metrics.errors?.rate || 0).toFixed(2)}%
            </p>
          </div>
        </div>
      </div>

      {/* Uptime */}
      <div className="bg-gray-700/50 rounded p-3">
        <p className="text-gray-400 text-sm">System Uptime</p>
        <p className="text-white text-xl font-medium">
          {formatDuration(metrics.uptime || 0)}
        </p>
        <p className="text-gray-400 text-xs">
          Since {new Date(metrics.timestamp).toLocaleString()}
        </p>
      </div>
    </div>
  );
};

// Helper function to format duration
const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  return `${hours}h ${minutes}m ${secs}s`;
};

export default PerformanceMonitor;
