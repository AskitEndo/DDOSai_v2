import React, {
  createContext,
  useContext,
  useReducer,
  useCallback,
  ReactNode,
} from "react";
import {
  DetectionResult,
  SystemMetrics,
  NetworkGraph,
  SimulationStatus,
} from "../types";

// Centralized data management to prevent loading/unloading conflicts
interface DataState {
  detections: DetectionResult[];
  metrics: SystemMetrics | null;
  networkGraph: NetworkGraph | null;
  simulation: SimulationStatus | null;
  isLoading: boolean;
  isConnected: boolean;
  usingDummyData: boolean;
  lastUpdated: {
    detections: Date | null;
    metrics: Date | null;
    networkGraph: Date | null;
    simulation: Date | null;
  };
}

type DataAction =
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_CONNECTION"; payload: boolean }
  | { type: "SET_DETECTIONS"; payload: DetectionResult[] }
  | { type: "ADD_DETECTION"; payload: DetectionResult }
  | { type: "SET_METRICS"; payload: SystemMetrics }
  | { type: "SET_NETWORK_GRAPH"; payload: NetworkGraph }
  | { type: "SET_SIMULATION"; payload: SimulationStatus }
  | {
      type: "SET_DUMMY_DATA";
      payload: {
        detections: DetectionResult[];
        metrics: SystemMetrics;
        networkGraph: NetworkGraph;
      };
    }
  | {
      type: "SET_BACKEND_DATA";
      payload: {
        detections?: DetectionResult[];
        metrics?: SystemMetrics;
        networkGraph?: NetworkGraph;
      };
    }
  | { type: "CLEAR_ALL_DATA" }
  | { type: "RESET_STATE" };

const initialState: DataState = {
  detections: [],
  metrics: null,
  networkGraph: null,
  simulation: null,
  isLoading: false,
  isConnected: false,
  usingDummyData: false,
  lastUpdated: {
    detections: null,
    metrics: null,
    networkGraph: null,
    simulation: null,
  },
};

const dataReducer = (state: DataState, action: DataAction): DataState => {
  const now = new Date();

  switch (action.type) {
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };

    case "SET_CONNECTION":
      return { ...state, isConnected: action.payload };

    case "SET_DETECTIONS":
      return {
        ...state,
        detections: action.payload,
        lastUpdated: { ...state.lastUpdated, detections: now },
      };

    case "ADD_DETECTION":
      return {
        ...state,
        detections: [action.payload, ...state.detections].slice(0, 1000),
        lastUpdated: { ...state.lastUpdated, detections: now },
      };

    case "SET_METRICS":
      return {
        ...state,
        metrics: action.payload,
        lastUpdated: { ...state.lastUpdated, metrics: now },
      };

    case "SET_NETWORK_GRAPH":
      return {
        ...state,
        networkGraph: action.payload,
        lastUpdated: { ...state.lastUpdated, networkGraph: now },
      };

    case "SET_SIMULATION":
      return {
        ...state,
        simulation: action.payload,
        lastUpdated: { ...state.lastUpdated, simulation: now },
      };

    case "SET_DUMMY_DATA":
      return {
        ...state,
        detections: action.payload.detections,
        metrics: action.payload.metrics,
        networkGraph: action.payload.networkGraph,
        usingDummyData: true,
        lastUpdated: {
          detections: now,
          metrics: now,
          networkGraph: now,
          simulation: state.lastUpdated.simulation,
        },
      };

    case "SET_BACKEND_DATA":
      return {
        ...state,
        ...(action.payload.detections && {
          detections: action.payload.detections,
        }),
        ...(action.payload.metrics && { metrics: action.payload.metrics }),
        ...(action.payload.networkGraph && {
          networkGraph: action.payload.networkGraph,
        }),
        usingDummyData: false,
        lastUpdated: {
          ...state.lastUpdated,
          ...(action.payload.detections && { detections: now }),
          ...(action.payload.metrics && { metrics: now }),
          ...(action.payload.networkGraph && { networkGraph: now }),
        },
      };

    case "CLEAR_ALL_DATA":
      return {
        ...initialState,
        isConnected: state.isConnected,
      };

    case "RESET_STATE":
      return initialState;

    default:
      return state;
  }
};

interface DataContextType {
  state: DataState;
  dispatch: React.Dispatch<DataAction>;
  loadDummyData: () => void;
  loadBackendData: () => Promise<void>;
  clearData: () => void;
  isDataStale: (
    type: keyof DataState["lastUpdated"],
    maxAgeMs?: number
  ) => boolean;
}

const DataContext = createContext<DataContextType | null>(null);

export const DataProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [state, dispatch] = useReducer(dataReducer, initialState);

  const loadDummyData = useCallback(() => {
    // Import here to avoid circular dependencies
    import("../utils/dummyData").then(
      ({
        generateRandomDetections,
        generateRandomMetrics,
        generateRandomNetworkGraph,
      }) => {
        const detections = generateRandomDetections(50);
        const metrics = generateRandomMetrics();
        const networkGraph = generateRandomNetworkGraph();

        dispatch({
          type: "SET_DUMMY_DATA",
          payload: { detections, metrics, networkGraph },
        });
      }
    );
  }, []);

  const loadBackendData = useCallback(async () => {
    dispatch({ type: "SET_LOADING", payload: true });
    try {
      // Import API service to avoid circular dependencies
      const api = (await import("../services/api")).default;

      const [
        metricsResponse,
        detectionsResponse,
        networkResponse,
        monitoringResponse,
      ] = await Promise.allSettled([
        api.getSystemMetrics(),
        api.getDetections(50),
        api.getNetworkGraph(),
        api.getNetworkMonitoringData(),
      ]);

      const backendData: {
        detections?: DetectionResult[];
        metrics?: SystemMetrics;
        networkGraph?: NetworkGraph;
      } = {};

      // Process metrics
      if (
        metricsResponse.status === "fulfilled" &&
        metricsResponse.value.status === 200
      ) {
        backendData.metrics = metricsResponse.value.data;
      }

      // Process detections
      if (
        detectionsResponse.status === "fulfilled" &&
        detectionsResponse.value.status === 200
      ) {
        backendData.detections = detectionsResponse.value.data;
      }

      // Process network graph
      if (
        networkResponse.status === "fulfilled" &&
        networkResponse.value.status === 200
      ) {
        backendData.networkGraph = networkResponse.value.data;
      }

      // Process monitoring data
      if (
        monitoringResponse.status === "fulfilled" &&
        monitoringResponse.value.status === 200
      ) {
        const monitoringData = monitoringResponse.value.data;

        // If monitoring is active and has detected attacks, add them to detections
        if (
          monitoringData.monitoring_active &&
          monitoringData.detected_attacks.length > 0
        ) {
          const networkDetections = monitoringData.detected_attacks.map(
            (attack: any) => ({
              timestamp: attack.timestamp,
              packet_id: `network_${Date.now()}_${Math.random()}`,
              flow_id: `flow_${attack.source_ip}_${attack.destination_ip}`,
              src_ip: attack.source_ip,
              dst_ip: attack.destination_ip,
              src_port: 0,
              dst_port: 0,
              protocol: attack.protocol,
              is_malicious: attack.is_malicious,
              threat_score: attack.confidence * 100,
              attack_type: attack.attack_type,
              detection_method: "network_monitor",
              confidence: attack.confidence,
              explanation: {
                source: "network_monitor",
                flags: attack.flags,
                packet_size: attack.packet_size,
                severity: attack.severity,
              },
              model_version: "network_monitor_v1.0",
            })
          );

          // Merge with existing detections
          backendData.detections = [
            ...(backendData.detections || []),
            ...networkDetections,
          ];
        }
      }

      dispatch({
        type: "SET_BACKEND_DATA",
        payload: backendData,
      });

      dispatch({ type: "SET_CONNECTION", payload: true });
    } catch (error) {
      console.error("Failed to load backend data:", error);
      dispatch({ type: "SET_CONNECTION", payload: false });
      throw error;
    } finally {
      dispatch({ type: "SET_LOADING", payload: false });
    }
  }, []);

  const clearData = useCallback(() => {
    dispatch({ type: "CLEAR_ALL_DATA" });
  }, []);

  const isDataStale = useCallback(
    (type: keyof DataState["lastUpdated"], maxAgeMs = 30000) => {
      const lastUpdate = state.lastUpdated[type];
      if (!lastUpdate) return true;
      return Date.now() - lastUpdate.getTime() > maxAgeMs;
    },
    [state.lastUpdated]
  );

  return (
    <DataContext.Provider
      value={{
        state,
        dispatch,
        loadDummyData,
        loadBackendData,
        clearData,
        isDataStale,
      }}
    >
      {children}
    </DataContext.Provider>
  );
};

export const useDataManager = () => {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error("useDataManager must be used within a DataProvider");
  }
  return context;
};
