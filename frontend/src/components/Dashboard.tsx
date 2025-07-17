import React, { useEffect, useState } from "react";
import { useAppContext } from "../context/AppContext";
import { useWebSocket } from "../hooks/useWebSocket";
import { useApi } from "../hooks/useApi";
import api from "../services/api";
import {
  DetectionResult,
  SystemMetrics,
  NetworkGraph as NetworkGraphType,
} from "../types";
import { formatNumber, getThreatLevelColor } from "../utils";
import { Activity, Shield, AlertTriangle, Network } from "lucide-react";
import NetworkGraph from "./NetworkGraph";
import LiveTrafficMonitor from "./LiveTrafficMonitor";
import ThreatScorePanel from "./ThreatScorePanel";
import MetricsPanel from "./MetricsPanel";
import XAIPanel from "./XAIPanel";
import DetectionDetails from "./DetectionDetails";
import PerformanceMonitor from "./PerformanceMonitor";

const Dashboard: React.FC = () => {
  const { state, dispatch } = useAppContext();

  // API hooks for fetching data
  const {
    data: detections,
    loading: detectionsLoading,
    refresh: refreshDetections,
  } = useApi(() => api.getDetections(50), {
    immediate: true,
    refreshInterval: 5000,
  });

  const {
    data: metrics,
    loading: metricsLoading,
    refresh: refreshMetrics,
  } = useApi(() => api.getSystemMetrics(), {
    immediate: true,
    refreshInterval: 3000,
  });

  const {
    data: networkGraph,
    loading: graphLoading,
    refresh: refreshGraph,
  } = useApi(() => api.getNetworkGraph(), {
    immediate: true,
    refreshInterval: 10000,
  });

  // WebSocket hooks for real-time updates
  useWebSocket("detection", (data: DetectionResult) => {
    dispatch({ type: "ADD_DETECTION", payload: data });
  });

  useWebSocket("metrics", (data: SystemMetrics) => {
    dispatch({ type: "SET_METRICS", payload: data });
  });

  useWebSocket("graph", (data: NetworkGraphType) => {
    dispatch({ type: "SET_NETWORK_GRAPH", payload: data });
  });

  // Update state when API data changes
  useEffect(() => {
    if (detections?.data) {
      dispatch({ type: "SET_DETECTIONS", payload: detections.data });
    }
  }, [detections, dispatch]);

  useEffect(() => {
    if (metrics?.data) {
      dispatch({ type: "SET_METRICS", payload: metrics.data });
    }
  }, [metrics, dispatch]);

  useEffect(() => {
    if (networkGraph?.data) {
      dispatch({ type: "SET_NETWORK_GRAPH", payload: networkGraph.data });
    }
  }, [networkGraph, dispatch]);

  const maliciousCount = state.detections.filter((d) => d.is_malicious).length;

  const [selectedDetection, setSelectedDetection] =
    useState<DetectionResult | null>(null);

  // Handle node click in network graph
  const handleNodeClick = (node: any) => {
    // Find a detection related to this node
    const nodeDetection = state.detections.find(
      (d) => d.src_ip === node.ip_address || d.dst_ip === node.ip_address
    );
    if (nodeDetection) {
      setSelectedDetection(nodeDetection);
    }
  };

  // Handle detection selection
  const handleSelectDetection = (detection: DetectionResult) => {
    setSelectedDetection(detection);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
        <div className="flex items-center space-x-4">
          <button
            onClick={() => {
              refreshDetections();
              refreshMetrics();
              refreshGraph();
            }}
            className="px-4 py-2 bg-primary/20 text-primary border border-primary/30 rounded-md hover:bg-primary/30 transition-colors"
          >
            <Activity className="w-4 h-4 mr-2 inline" />
            Refresh
          </button>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Packets"
          value={state.metrics?.packets_processed || 0}
          icon={<Activity className="w-6 h-6" />}
          color="text-primary"
          loading={metricsLoading}
        />
        <MetricCard
          title="Threats Detected"
          value={maliciousCount}
          icon={<Shield className="w-6 h-6" />}
          color="text-danger"
          loading={detectionsLoading}
        />
        <MetricCard
          title="Threat Level"
          value={`${state.metrics?.threat_level || 0}/5`}
          icon={<AlertTriangle className="w-6 h-6" />}
          color={getThreatLevelColor((state.metrics?.threat_level || 0) * 20)}
          loading={metricsLoading}
        />
        <MetricCard
          title="Active Connections"
          value={state.metrics?.active_connections || 0}
          icon={<Network className="w-6 h-6" />}
          color="text-info"
          loading={metricsLoading}
        />
      </div>

      {/* Main Dashboard Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          {/* Live Traffic Monitor */}
          <LiveTrafficMonitor detections={state.detections} />

          {/* Threat Score Panel */}
          <ThreatScorePanel
            detections={state.detections}
            loading={detectionsLoading}
            onSelectDetection={handleSelectDetection}
          />
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          {/* Metrics Panel */}
          <MetricsPanel metrics={state.metrics} loading={metricsLoading} />

          {/* XAI Panel */}
          <XAIPanel detection={selectedDetection} loading={false} />
        </div>
      </div>

      {/* Network Graph (Full Width) */}
      <div className="w-full">
        <NetworkGraph
          data={state.networkGraph}
          loading={graphLoading}
          height={500}
          onNodeClick={handleNodeClick}
          onRefresh={refreshGraph}
        />
      </div>

      {/* Detection Details Modal */}
      {state.selectedDetection && (
        <DetectionDetails
          detection={state.selectedDetection}
          onClose={() =>
            dispatch({ type: "SET_SELECTED_DETECTION", payload: null })
          }
        />
      )}
    </div>
  );
};

// Metric Card Component
interface MetricCardProps {
  title: string;
  value: number | string;
  icon: React.ReactNode;
  color: string;
  loading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  icon,
  color,
  loading,
}) => {
  return (
    <div className="bg-dark-surface border border-dark-border rounded-lg p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm font-medium">{title}</p>
          {loading ? (
            <div className="animate-pulse bg-gray-600 h-8 w-16 rounded mt-2"></div>
          ) : (
            <p className="text-2xl font-bold text-white mt-1">
              {typeof value === "number" ? formatNumber(value) : value}
            </p>
          )}
        </div>
        <div className={`${color} opacity-80`}>{icon}</div>
      </div>
    </div>
  );
};

export default Dashboard;
