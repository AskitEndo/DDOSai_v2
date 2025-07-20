import React, {
  createContext,
  useContext,
  useReducer,
  useCallback,
  ReactNode,
} from "react";
import { SimulationConfig, SimulationStatus } from "../types";
import api from "../services/api";

interface SimulationState {
  currentSimulation: SimulationStatus | null;
  isRunning: boolean;
  lastConfig: SimulationConfig | null;
  userIP: string | null;
  history: SimulationStatus[];
}

type SimulationAction =
  | { type: "SET_SIMULATION"; payload: SimulationStatus }
  | { type: "CLEAR_SIMULATION" }
  | { type: "SET_RUNNING"; payload: boolean }
  | { type: "SET_CONFIG"; payload: SimulationConfig }
  | { type: "SET_USER_IP"; payload: string }
  | { type: "ADD_TO_HISTORY"; payload: SimulationStatus };

const initialState: SimulationState = {
  currentSimulation: null,
  isRunning: false,
  lastConfig: null,
  userIP: null,
  history: [],
};

const simulationReducer = (
  state: SimulationState,
  action: SimulationAction
): SimulationState => {
  switch (action.type) {
    case "SET_SIMULATION":
      return {
        ...state,
        currentSimulation: action.payload,
        isRunning: action.payload.status === "running",
      };

    case "CLEAR_SIMULATION":
      return {
        ...state,
        currentSimulation: null,
        isRunning: false,
      };

    case "SET_RUNNING":
      return {
        ...state,
        isRunning: action.payload,
      };

    case "SET_CONFIG":
      return {
        ...state,
        lastConfig: action.payload,
      };

    case "SET_USER_IP":
      return {
        ...state,
        userIP: action.payload,
      };

    case "ADD_TO_HISTORY":
      return {
        ...state,
        history: [action.payload, ...state.history].slice(0, 10), // Keep last 10
      };

    default:
      return state;
  }
};

interface SimulationContextType {
  state: SimulationState;
  dispatch: React.Dispatch<SimulationAction>;
  startSimulation: (config: SimulationConfig) => Promise<boolean>;
  stopSimulation: () => Promise<boolean>;
  forceStopSimulation: () => Promise<boolean>;
  getUserIP: () => Promise<string>;
  refreshSimulationStatus: () => Promise<void>;
}

const SimulationContext = createContext<SimulationContextType | null>(null);

export const SimulationProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [state, dispatch] = useReducer(simulationReducer, initialState);

  const getUserIP = useCallback(async (): Promise<string> => {
    try {
      // Try to get IP from a reliable service
      const response = await fetch("https://api.ipify.org?format=json");
      const data = await response.json();
      const userIP = data.ip;
      dispatch({ type: "SET_USER_IP", payload: userIP });
      return userIP;
    } catch (error) {
      console.error("Failed to get user IP:", error);
      // Fallback to local detection or default
      const fallbackIP = "192.168.1.100"; // Default for demo
      dispatch({ type: "SET_USER_IP", payload: fallbackIP });
      return fallbackIP;
    }
  }, []);

  const startSimulation = useCallback(
    async (config: SimulationConfig): Promise<boolean> => {
      try {
        dispatch({ type: "SET_CONFIG", payload: config });
        dispatch({ type: "SET_RUNNING", payload: true });

        const response = await api.startSimulation(config);

        if (response.status === 200) {
          const simulationStatus: SimulationStatus = {
            simulation_id: response.data.simulation_id || `sim_${Date.now()}`,
            status: "running",
            attack_type: config.attack_type,
            target_ip: config.target_ip,
            target_port: config.target_port,
            duration: config.duration,
            packet_rate: config.packet_rate,
            packets_sent: 0,
            bytes_sent: 0,
            errors: 0,
            start_time: new Date().toISOString(),
            end_time: undefined,
          };

          dispatch({ type: "SET_SIMULATION", payload: simulationStatus });
          dispatch({ type: "ADD_TO_HISTORY", payload: simulationStatus });
          return true;
        } else {
          dispatch({ type: "SET_RUNNING", payload: false });
          return false;
        }
      } catch (error) {
        console.error("Failed to start simulation:", error);
        dispatch({ type: "SET_RUNNING", payload: false });
        return false;
      }
    },
    []
  );

  const stopSimulation = useCallback(async (): Promise<boolean> => {
    try {
      if (state.currentSimulation?.simulation_id) {
        const response = await api.stopSimulation(
          state.currentSimulation.simulation_id
        );

        if (response.status === 200) {
          const updatedSimulation: SimulationStatus = {
            ...state.currentSimulation,
            status: "completed",
            end_time: new Date().toISOString(),
          };

          dispatch({ type: "SET_SIMULATION", payload: updatedSimulation });
          dispatch({ type: "ADD_TO_HISTORY", payload: updatedSimulation });
          dispatch({ type: "SET_RUNNING", payload: false });
          return true;
        }
      }

      dispatch({ type: "SET_RUNNING", payload: false });
      return false;
    } catch (error) {
      console.error("Failed to stop simulation:", error);
      dispatch({ type: "SET_RUNNING", payload: false });
      return false;
    }
  }, [state.currentSimulation]);

  const refreshSimulationStatus = useCallback(async () => {
    try {
      // Check if backend is reachable first
      const healthResponse = await api.health();
      if (healthResponse.status !== 200) {
        // Backend is down, clear simulation state
        dispatch({ type: "CLEAR_SIMULATION" });
        dispatch({ type: "SET_RUNNING", payload: false });
        return;
      }

      // Check actual simulation status from backend
      const statusResponse = await api.getAllSimulationStatus();
      if (statusResponse.status === 200) {
        const backendStatus = statusResponse.data;

        // If no active simulations on backend but frontend thinks there is one
        if (backendStatus.active_simulations === 0 && state.isRunning) {
          console.log(
            "Backend has no active simulations, clearing frontend state"
          );
          dispatch({ type: "CLEAR_SIMULATION" });
          dispatch({ type: "SET_RUNNING", payload: false });
        }

        // Update simulation state based on backend reality
        if (state.currentSimulation && backendStatus.active_simulations > 0) {
          const updatedSimulation = {
            ...state.currentSimulation,
            status: "running" as const,
          };
          dispatch({ type: "SET_SIMULATION", payload: updatedSimulation });
        }
      }
    } catch (error) {
      console.error(
        "Failed to refresh simulation status, clearing state:",
        error
      );
      // If we can't reach backend, clear simulation state
      dispatch({ type: "CLEAR_SIMULATION" });
      dispatch({ type: "SET_RUNNING", payload: false });
    }
  }, [state.currentSimulation, state.isRunning]);

  const forceStopSimulation = useCallback(async () => {
    try {
      // Force clear simulation state regardless of backend response
      dispatch({ type: "CLEAR_SIMULATION" });
      dispatch({ type: "SET_RUNNING", payload: false });

      // Try to notify backend if possible
      if (state.currentSimulation?.simulation_id) {
        try {
          await api.stopSimulation(state.currentSimulation.simulation_id);
        } catch (error) {
          console.log(
            "Backend not reachable for stop notification, state cleared anyway"
          );
        }
      }

      return true;
    } catch (error) {
      console.error("Force stop failed:", error);
      return false;
    }
  }, [state.currentSimulation]);

  return (
    <SimulationContext.Provider
      value={{
        state,
        dispatch,
        startSimulation,
        stopSimulation,
        forceStopSimulation,
        getUserIP,
        refreshSimulationStatus,
      }}
    >
      {children}
    </SimulationContext.Provider>
  );
};

export const useSimulation = () => {
  const context = useContext(SimulationContext);
  if (!context) {
    throw new Error("useSimulation must be used within a SimulationProvider");
  }
  return context;
};
