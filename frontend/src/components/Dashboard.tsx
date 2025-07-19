import React, { useEffect, useState } from "react";
import { useAppContext } from "../context/AppContext";
import { useDataManager } from "../context/DataManager";
import webSocketService from "../services/websocket";
import { useApi } from "../hooks/useApi";
import api from "../services/api";
import {
  DetectionResult,
  SystemMetrics,
  NetworkGraph as NetworkGraphType,
} from "../types";
import { formatNumber, getThreatLevelColor } from "../utils";
import {
  generateRandomDetections,
  generateRandomMetrics,
  generateRandomNetworkGraph,
} from "../utils/dummyData";
import { Activity, Shield, AlertTriangle, Network } from "lucide-react";
import NetworkGraph from "./NetworkGraph";
import LiveTrafficMonitor from "./LiveTrafficMonitor";
import ThreatScorePanel from "./ThreatScorePanel";
import MetricsPanel from "./MetricsPanel";
import XAIPanel from "./XAIPanel";
import DetectionDetails from "./DetectionDetails";
import DataControls from "./DataControls";
import NetworkMonitoring from "./NetworkMonitoring";

const Dashboard: React.FC = () => {
  const { state: appState, dispatch: appDispatch } = useAppContext();
  const { state: dataState } = useDataManager();
  const [selectedDetection, setSelectedDetection] =
    useState<DetectionResult | null>(null);

  // Check if we're in offline mode - note that we consider the app "connected" even in offline mode
  // because we're using dummy data, so we check localStorage directly
  const isOfflineMode = localStorage.getItem("ddosai_offline_mode") === "true";

  // Remove automatic data loading - let user click "Load Data" instead
  // useEffect(() => {
  //   if (
  //     (!state.detections || state.detections.length === 0) &&
  //     !state.metrics &&
  //     !state.networkGraph
  //   ) {
  //     console.log("No data available, loading sample data automatically");
  //     // Generate random data
  //     const sampleDetections = generateRandomDetections(50);
  //     const sampleMetrics = generateRandomMetrics();
  //     const sampleNetworkGraph = generateRandomNetworkGraph();

  //     // Update the app state with dummy data immediately to prevent flickering
  //     dispatch({ type: "SET_DETECTIONS", payload: sampleDetections });
  //     dispatch({ type: "SET_METRICS", payload: sampleMetrics });
  //     dispatch({ type: "SET_NETWORK_GRAPH", payload: sampleNetworkGraph });
  //   }
  // }, [state.detections, state.metrics, state.networkGraph, dispatch]);

  // API hooks for fetching data - disabled by default, only enabled when data is loaded
  const {
    data: detections,
    loading: detectionsLoading,
    refresh: refreshDetections,
  } = useApi(() => api.getDetections(50), {
    immediate: false, // Don't auto-start API calls
    refreshInterval: 0, // Disable auto-refresh, let user control it
  });

  const {
    data: metrics,
    loading: metricsLoading,
    refresh: refreshMetrics,
  } = useApi(() => api.getSystemMetrics(), {
    immediate: false, // Don't auto-start API calls
    refreshInterval: 0, // Disable auto-refresh, let user control it
  });

  const {
    data: networkGraph,
    loading: graphLoading,
    refresh: refreshGraph,
  } = useApi(() => api.getNetworkGraph(), {
    immediate: false, // Don't auto-start API calls
    refreshInterval: 0, // Disable auto-refresh, let user control it
  });

  // In offline mode, we don't want to show loading states
  const actualDetectionsLoading = isOfflineMode ? false : detectionsLoading;
  const actualMetricsLoading = isOfflineMode ? false : metricsLoading;
  const actualGraphLoading = isOfflineMode ? false : graphLoading;

  // WebSocket hooks for real-time updates - only if not in offline mode
  useEffect(() => {
    if (!isOfflineMode) {
      // Only set up WebSocket listeners if not in offline mode
      const handleDetection = (data: DetectionResult) => {
        appDispatch({ type: "ADD_DETECTION", payload: data });
      };

      const handleMetrics = (data: SystemMetrics) => {
        appDispatch({ type: "SET_METRICS", payload: data });
      };

      const handleGraph = (data: NetworkGraphType) => {
        appDispatch({ type: "SET_NETWORK_GRAPH", payload: data });
      };

      // Subscribe to WebSocket events
      webSocketService.on("detection", handleDetection);
      webSocketService.on("metrics", handleMetrics);
      webSocketService.on("graph", handleGraph);

      return () => {
        // Clean up WebSocket listeners
        webSocketService.off("detection", handleDetection);
        webSocketService.off("metrics", handleMetrics);
        webSocketService.off("graph", handleGraph);
      };
    }
  }, [isOfflineMode, appDispatch]);

  // Update state when API data changes - only if not in offline mode
  useEffect(() => {
    if (!isOfflineMode && detections?.data) {
      appDispatch({ type: "SET_DETECTIONS", payload: detections.data });
    }
  }, [detections, appDispatch, isOfflineMode]);

  useEffect(() => {
    if (!isOfflineMode && metrics?.data) {
      appDispatch({ type: "SET_METRICS", payload: metrics.data });
    }
  }, [metrics, appDispatch, isOfflineMode]);

  useEffect(() => {
    if (!isOfflineMode && networkGraph?.data) {
      appDispatch({ type: "SET_NETWORK_GRAPH", payload: networkGraph.data });
    }
  }, [networkGraph, appDispatch, isOfflineMode]);

  const maliciousCount = Array.isArray(dataState.detections)
    ? dataState.detections.filter((d: DetectionResult) => d.is_malicious).length
    : 0;

  // Handle node click in network graph
  const handleNodeClick = (node: any) => {
    // Find a detection related to this node
    if (Array.isArray(dataState.detections)) {
      const nodeDetection = dataState.detections.find(
        (d: DetectionResult) =>
          d.src_ip === node.ip_address || d.dst_ip === node.ip_address
      );
      if (nodeDetection) {
        setSelectedDetection(nodeDetection);
      }
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
          <DataControls showBoth={true} />
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
          value={dataState.metrics?.packets_processed || 0}
          icon={<Activity className="w-6 h-6" />}
          color="text-primary"
          loading={actualMetricsLoading}
        />
        <MetricCard
          title="Threats Detected"
          value={maliciousCount}
          icon={<Shield className="w-6 h-6" />}
          color="text-danger"
          loading={actualDetectionsLoading}
        />
        <MetricCard
          title="Threat Level"
          value={`${dataState.metrics?.threat_level || 0}/5`}
          icon={<AlertTriangle className="w-6 h-6" />}
          color={getThreatLevelColor(
            (dataState.metrics?.threat_level || 0) * 20
          )}
          loading={actualMetricsLoading}
        />
        <MetricCard
          title="Active Connections"
          value={dataState.metrics?.active_connections || 0}
          icon={<Network className="w-6 h-6" />}
          color="text-info"
          loading={actualMetricsLoading}
        />
      </div>

      {/* Network Monitoring Panel */}
      <NetworkMonitoring className="w-full" />

      {/* Main Dashboard Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          {/* Live Traffic Monitor */}
          <LiveTrafficMonitor detections={dataState.detections} />

          {/* Threat Score Panel */}
          <ThreatScorePanel
            detections={dataState.detections}
            loading={actualDetectionsLoading}
            onSelectDetection={handleSelectDetection}
          />
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          {/* Metrics Panel */}
          <MetricsPanel
            metrics={dataState.metrics}
            loading={actualMetricsLoading}
          />

          {/* XAI Panel */}
          <XAIPanel detection={selectedDetection} loading={false} />
        </div>
      </div>

      {/* Network Graph (Full Width) */}
      <div className="w-full">
        <NetworkGraph
          data={dataState.networkGraph}
          loading={actualGraphLoading}
          height={500}
          onNodeClick={handleNodeClick}
          onRefresh={refreshGraph}
        />
      </div>

      {/* Detection Details Modal */}
      {selectedDetection && (
        <DetectionDetails
          detection={selectedDetection}
          onClose={() => setSelectedDetection(null)}
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
    <div className="bg-gray-800/70 border border-gray-700 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:border-blue-700/50 hover:bg-gray-800/90">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm font-medium">{title}</p>
          {loading ? (
            <div className="animate-pulse bg-gray-700 h-8 w-24 rounded mt-2"></div>
          ) : (
            <p className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-blue-200 mt-1">
              {typeof value === "number" ? formatNumber(value) : value}
            </p>
          )}
        </div>
        <div className={`p-3 rounded-full bg-gray-700/50 ${color}`}>{icon}</div>
      </div>
    </div>
  );
};

export default Dashboard;
