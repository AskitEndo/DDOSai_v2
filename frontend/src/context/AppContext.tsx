import React, { createContext, useContext, useReducer, ReactNode } from "react";
import {
  DetectionResult,
  SystemMetrics,
  NetworkGraph,
  SimulationStatus,
  UserSettings,
} from "../types";

// State interface
interface AppState {
  // System data
  detections: DetectionResult[];
  metrics: SystemMetrics | null;
  networkGraph: NetworkGraph | null;

  // Simulation state
  simulation: SimulationStatus | null;

  // UI state
  isConnected: boolean;
  alertsEnabled: boolean;
  selectedDetection: DetectionResult | null;

  // User settings
  settings: UserSettings;
}

// Action types
type AppAction =
  | { type: "SET_DETECTIONS"; payload: DetectionResult[] }
  | { type: "ADD_DETECTION"; payload: DetectionResult }
  | { type: "SET_METRICS"; payload: SystemMetrics }
  | { type: "SET_NETWORK_GRAPH"; payload: NetworkGraph }
  | { type: "SET_SIMULATION"; payload: SimulationStatus }
  | { type: "SET_CONNECTION_STATUS"; payload: boolean }
  | { type: "SET_ALERTS_ENABLED"; payload: boolean }
  | { type: "SET_SELECTED_DETECTION"; payload: DetectionResult | null }
  | { type: "UPDATE_SETTINGS"; payload: Partial<UserSettings> }
  | { type: "CLEAR_DATA" };

// Initial state
const initialState: AppState = {
  detections: [],
  metrics: null,
  networkGraph: null,
  simulation: null,
  isConnected: false,
  alertsEnabled: true,
  selectedDetection: null,
  settings: {
    theme: "dark",
    refreshRate: 5000,
    alertSound: true,
    dashboardLayout: [
      {
        id: "traffic",
        title: "Live Traffic",
        type: "traffic",
        width: "half",
        height: "medium",
      },
      {
        id: "threats",
        title: "Threat Detection",
        type: "threats",
        width: "half",
        height: "medium",
      },
      {
        id: "graph",
        title: "Network Graph",
        type: "graph",
        width: "full",
        height: "large",
      },
      {
        id: "metrics",
        title: "System Metrics",
        type: "metrics",
        width: "third",
        height: "small",
      },
    ],
  },
};

// Reducer function
const appReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case "SET_DETECTIONS":
      return { ...state, detections: action.payload };

    case "ADD_DETECTION":
      return {
        ...state,
        detections: [action.payload, ...state.detections].slice(0, 1000), // Keep last 1000
      };

    case "SET_METRICS":
      return { ...state, metrics: action.payload };

    case "SET_NETWORK_GRAPH":
      return { ...state, networkGraph: action.payload };

    case "SET_SIMULATION":
      return { ...state, simulation: action.payload };

    case "SET_CONNECTION_STATUS":
      return { ...state, isConnected: action.payload };

    case "SET_ALERTS_ENABLED":
      return { ...state, alertsEnabled: action.payload };

    case "SET_SELECTED_DETECTION":
      return { ...state, selectedDetection: action.payload };

    case "UPDATE_SETTINGS":
      return {
        ...state,
        settings: { ...state.settings, ...action.payload },
      };

    case "CLEAR_DATA":
      return {
        ...state,
        detections: [],
        metrics: null,
        networkGraph: null,
        selectedDetection: null,
      };

    default:
      return state;
  }
};

// Context
const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
} | null>(null);

// Provider component
export const AppProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};

// Custom hook to use the context
export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useAppContext must be used within an AppProvider");
  }
  return context;
};

export default AppContext;
