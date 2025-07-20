import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from "axios";
import {
  ApiResponse,
  DetectionResult,
  NetworkGraph,
  SystemMetrics,
  SimulationConfig,
  SimulationStatus,
} from "../types";

// Create axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 10000,
});

// Add request interceptor for authentication if needed
apiClient.interceptors.request.use(
  (config) => {
    // You can add auth token here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    // Clear offline mode flag if we get a successful response
    localStorage.removeItem("ddosai_offline_mode");
    return response;
  },
  (error) => {
    console.error("API Error:", error);

    // Check if this is a network error (likely offline mode)
    if (error.code === "ERR_NETWORK") {
      console.log("Network error detected, setting offline mode");
      // Store offline status in localStorage to persist across page reloads
      localStorage.setItem("ddosai_offline_mode", "true");
    }

    return Promise.reject(error);
  }
);

// Generic request method
const request = async <T>(
  config: AxiosRequestConfig
): Promise<ApiResponse<T>> => {
  try {
    const response: AxiosResponse = await apiClient(config);
    return {
      data: response.data,
      status: response.status,
    };
  } catch (error: any) {
    if (error.response) {
      return {
        data: error.response.data,
        status: error.response.status,
        message: error.response.data.message || "An error occurred",
      };
    }
    return {
      data: {} as T,
      status: 500,
      message: error.message || "Network error",
    };
  }
};

// API endpoints
export const api = {
  // Health check
  health: () => request<{ status: string }>({ method: "GET", url: "/health" }),

  // Traffic analysis
  analyzeTraffic: (packet: any) =>
    request<DetectionResult>({
      method: "POST",
      url: "/api/analyze",
      data: packet,
    }),

  // Detection results
  getDetections: (limit: number = 50) =>
    request<DetectionResult[]>({
      method: "GET",
      url: `/api/detections?limit=${limit}`,
    }),

  // Network graph
  getNetworkGraph: () =>
    request<NetworkGraph>({ method: "GET", url: "/api/graph/current" }),

  // System metrics
  getSystemMetrics: (detailed: boolean = false) =>
    request<SystemMetrics>({
      method: "GET",
      url: `/api/metrics${detailed ? "?detailed=true" : ""}`,
    }),

  // XAI explanations
  getExplanation: (predictionId: string) =>
    request<any>({ method: "GET", url: `/api/explain/${predictionId}` }),

  // Simulation control
  startSimulation: (config: SimulationConfig) =>
    request<{ simulation_id: string; status: string }>({
      method: "POST",
      url: "/api/simulate/start",
      data: config,
    }),

  stopSimulation: (simulationId: string) =>
    request<SimulationStatus>({
      method: "POST",
      url: `/api/simulate/stop?simulation_id=${simulationId}`,
    }),

  getSimulationStatus: (simulationId: string) =>
    request<SimulationStatus>({
      method: "GET",
      url: `/api/simulate/status?simulation_id=${simulationId}`,
    }),

  getAllSimulationStatus: () =>
    request<{
      active_simulations: number;
      simulations: any[];
      total_attacks_generated: number;
      recent_detections: number;
    }>({
      method: "GET",
      url: "/api/simulate/status",
    }),

  // Network monitoring for cross-device attacks
  getNetworkMonitoringData: () =>
    request<{
      monitoring_active: boolean;
      detected_attacks: any[];
      network_stats: any;
      total_monitored_packets: number;
      active_monitoring_duration: string;
      timestamp: string;
    }>({
      method: "GET",
      url: "/api/network/monitoring",
    }),

  startNetworkMonitoring: () =>
    request<{ status: string; message: string; monitoring_active: boolean }>({
      method: "POST",
      url: "/api/network/monitoring/start",
    }),

  stopNetworkMonitoring: () =>
    request<{ status: string; message: string; monitoring_active: boolean }>({
      method: "POST",
      url: "/api/network/monitoring/stop",
    }),
};

export default api;
